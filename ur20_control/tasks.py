"""Sorting actions invoked by the small UR20 state machine."""

import numpy as np
import omni.physx
from pxr import Usd, UsdGeom

from .geometry import authored_rigid_pose, transform
from .lula import TCP_FRAME
from .workcell import GUARD, PICKUP_BELT, ROUTES

RELEASE_CLEARANCE = 0.03  # Product bottom above the rollers before suction opens.


def physical_pose(path):
    """Read an active rigid body's world transform from PhysX."""
    value = omni.physx.get_physx_interface().get_rigidbody_transformation(str(path))
    if not value.get("ret_val"):
        raise RuntimeError("Physics body unavailable: " + str(path))
    x, y, z, w = value["rotation"]
    return transform(value["position"], [w, x, y, z])


class TaskExecutor:
    """Perform one product transaction using measured arm and product state."""

    def __init__(self, workcell, robot):
        """Capture the scene and controller used by every sorting action."""
        self.cell, self.robot = workcell, robot
        self.target = None
        self.phase = None
        self.elapsed = 0.0
        self.stable = 0.0
        self.rotation = np.diag([-1.0, 1.0, -1.0])

    def begin(self, path):
        """Validate a queued product and begin settling at the Guard."""
        product = self.cell.product(path)
        if not product or product.GetCustomDataByKey("ngv:sortedLane"):
            return False
        self.target = str(product.GetPath())
        self.lane = ROUTES[product.GetCustomDataByKey("ngv:product")][0]
        self.phase, self.elapsed, self.stable = "settle", 0.0, 0.0
        self.previous = physical_pose(self.target)[:3, 3]
        return True

    def _advance(self, phase):
        """Change the current task step and clear timing counters."""
        self.phase, self.elapsed, self.stable = phase, 0.0, 0.0
        self.robot.goal_key = None

    def _arrived(self, arrived, dt):
        """Require a target to stay reached briefly before changing steps."""
        self.stable = self.stable + dt if arrived else 0.0
        return self.stable >= 0.15

    def _product_bounds(self):
        """Transform local product corners using the measured PhysX pose."""
        corners = self.corners @ physical_pose(self.target).T
        return corners[:, :3].min(axis=0), corners[:, :3].max(axis=0)

    def _guard_clearance(self, point, rotation):
        """Measure the terminal suction housing's gap from the Guard."""
        joints = self.robot.model.pose_joints(point, rotation, self.robot.joints()[0])
        low, high = self.cell.guard_bounds
        return min(
            np.linalg.norm(center - np.clip(center, low, high)) - radius
            for center, radius in self.robot.model.collision_spheres_at(
                joints, TCP_FRAME
            )
        )

    def _prepare_grasp(self, product):
        """Choose an upright central suction pose that clears the Guard."""
        box = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
            .ComputeWorldBound(product)
            .ComputeAlignedRange()
        )
        low, high = np.asarray(box.GetMin()), np.asarray(box.GetMax())
        world_corners = np.array(
            [
                [x, y, z, 1]
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ]
        )
        authored_pose = authored_rigid_pose(product)
        # World bounds already include every authored scale. Remove only the
        # position/rotation that will be replaced by PhysX, keeping scaled size.
        self.corners = world_corners @ np.linalg.inv(authored_pose).T
        # Conveyor motion changes the PhysX body pose without rewriting its
        # authored USD transform, so grasp rays use the measured footprint.
        low, high = self._product_bounds()
        self.support_path = self.cell.support_collider(product, low, high)
        guard_low, guard_high = self.cell.guard_bounds
        center = (low[:2] + high[:2]) * 0.5
        guard_center = (guard_low[:2] + guard_high[:2]) * 0.5
        # A small offset inside the central top patch can clear the Guard
        # when a short carton stops against its face.
        points = [center]
        for distance in (0.02, 0.03, 0.04):
            for axis in (0, 1):
                shifted = center.copy()
                shifted[axis] += distance * np.sign(center[axis] - guard_center[axis])
                if low[axis] + 0.015 < shifted[axis] < high[axis] - 0.015:
                    points.append(shifted)
        best_clearance = -float("inf")
        for xy in points:
            try:
                point = self.cell.central_grasp(product, low, high, xy)
            except RuntimeError:
                continue
            candidates = []
            for yaw in (0, -90, 90, 180):
                angle = np.deg2rad(yaw)
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag(
                    [-1.0, 1.0, -1.0]
                )
                try:
                    clearance = self._guard_clearance(point, rotation)
                except RuntimeError:
                    continue
                candidates.append((clearance, rotation))
                best_clearance = max(best_clearance, clearance)
            valid = max(candidates, key=lambda item: item[0]) if candidates else None
            if valid is not None and valid[0] >= 0.01:
                self.pick_position = point
                self.rotation = valid[1]
                break
        else:
            raise RuntimeError(
                "No upright central grasp clears the Guard: "
                + f"best gap {best_clearance:.3f} m, "
                + f"product {low.tolist()}..{high.tolist()}, "
                + f"Guard {guard_low.tolist()}..{guard_high.tolist()}"
            )
        self.pick_local = (
            np.linalg.inv(physical_pose(self.target)) @ np.r_[self.pick_position, 1.0]
        )
        self.safe_z = max(1.55, self.pick_position[2] + 0.60)
        self.approach = self.pick_position.copy()
        self.approach[2] = self.safe_z

    def _track_pick(self):
        """Refresh the selected suction point while the carton settles."""
        self.pick_position = (physical_pose(self.target) @ self.pick_local)[:3]
        self.approach[:2] = self.pick_position[:2]

    def pick(self, dt):
        """Settle, approach, attach, and lift one product clear of the Guard."""
        self.elapsed += dt
        product = self.cell.product(self.target)
        if not product:
            raise RuntimeError("Target vanished before pickup")
        if self.phase == "settle":
            position = physical_pose(self.target)[:3, 3]
            speed = np.linalg.norm(position - self.previous) / max(dt, 1e-5)
            self.previous = position
            if self.elapsed > 5:
                raise RuntimeError("Product did not stop at Guard")
            if self.elapsed >= 0.6 and self._arrived(speed < 0.025, dt):
                self._prepare_grasp(product)
                self._advance("approach")
        elif self.phase == "approach":
            self._track_pick()
            if self._arrived(
                self.robot.move_pose(
                    self.approach, self.rotation, dt, target=self.target
                ),
                dt,
            ):
                self._advance("descend")
        elif self.phase == "descend":
            self._track_pick()
            if self._arrived(
                self.robot.move_pose(
                    self.pick_position,
                    self.rotation,
                    dt,
                    target=self.target,
                    allowed_contacts=(PICKUP_BELT,),
                    tolerance=0.02,
                ),
                dt,
            ):
                self.robot.suction(True)
                self._advance("attach")
        elif self.phase == "attach":
            if self.robot.holding(self.target):
                tcp = self.robot.tcp()
                self.tcp_to_product = np.linalg.inv(tcp) @ physical_pose(self.target)
                low, high = self._product_bounds()
                # Enclose the entire attached product in a TCP-frame sphere.
                local_low, local_high = (
                    self.corners[:, :3].min(axis=0),
                    self.corners[:, :3].max(axis=0),
                )
                local_center = (local_low + local_high) * 0.5
                payload_center = (self.tcp_to_product @ np.r_[local_center, 1.0])[:3]
                self.robot.set_payload(
                    payload_center, np.linalg.norm(local_high - local_low) * 0.5
                )
                self.bottom = low[2]
                # Project the complete scaled product at the commanded tool
                # orientation, including any measured suction pose offset.
                at_tcp = self.corners @ self.tcp_to_product.T
                placed = at_tcp[:, :3] @ self.rotation.T
                placed_low, placed_high = placed.min(axis=0), placed.max(axis=0)
                surface = self.cell.placement(
                    self.lane, (placed_high[:2] - placed_low[:2]) * 0.5
                )
                self.offset = -np.r_[
                    (placed_low[:2] + placed_high[:2]) * 0.5, placed_low[2]
                ]
                self.place = surface + self.offset + np.array(
                    [0, 0, RELEASE_CLEARANCE]
                )
                self.place_above = self.place.copy()
                self.place_above[2] = self.safe_z
                # A short escape from the Guard is the only fixed clearance step.
                self.clearance = tcp[:3, 3].copy()
                axis = int(
                    np.argmin(
                        self.cell.guard_bounds[1][:2] - self.cell.guard_bounds[0][:2]
                    )
                )
                direction = (
                    1
                    if (low[axis] + high[axis]) / 2
                    >= np.mean(
                        [
                            self.cell.guard_bounds[0][axis],
                            self.cell.guard_bounds[1][axis],
                        ]
                    )
                    else -1
                )
                self.clearance[axis] += direction * 0.10
                self.clearance[0] -= 0.10
                self.clearance[2] += 0.08
                self._advance("clear_guard")
            elif self.elapsed > 2:
                raise RuntimeError("Suction did not attach to target")
        elif self.phase == "clear_guard":
            self._require_hold()
            if self._arrived(
                self.robot.move_pose(
                    self.clearance,
                    self.rotation,
                    dt,
                    target=self.target,
                    allowed_contacts=(GUARD, PICKUP_BELT),
                ),
                dt,
            ):
                self._advance("lift")
        elif self.phase == "lift":
            self._require_hold()
            lift = self.clearance.copy()
            lift[2] = self.safe_z
            if self._arrived(
                self.robot.move_pose(
                    lift,
                    self.rotation,
                    dt,
                    target=self.target,
                    allowed_contacts=(GUARD, PICKUP_BELT),
                ),
                dt,
            ):
                if self._product_bounds()[0][2] < self.bottom + 0.55:
                    raise RuntimeError("Product did not rise with suction cup")
                return True
        if self.elapsed > 35:
            raise RuntimeError("Pickup motion timed out")
        return False

    def _require_hold(self):
        """Fail closed if the selected carton leaves the suction cup."""
        if not self.robot.holding(self.target):
            raise RuntimeError("Held product detached")

    def carry(self, dt):
        """Transport the held carton and verify release on its output lane."""
        self.elapsed += dt
        if self.phase == "lift":
            self._advance("transfer")
        if self.phase == "transfer":
            self._require_hold()
            if self._product_bounds()[0][2] < self.bottom + 0.55:
                raise RuntimeError("Product dropped below transfer clearance")
            if self._arrived(
                self.robot.move_pose(
                    self.place_above, self.rotation, dt, target=self.target
                ),
                dt,
            ):
                self._advance("lower")
        elif self.phase == "lower":
            self._require_hold()
            if self._arrived(
                self.robot.move_pose(
                    self.place,
                    self.rotation,
                    dt,
                    target=self.target,
                    allowed_lane=self.lane,
                    tolerance=0.008,
                ),
                dt,
            ):
                self.robot.suction(False)
                self.robot.set_payload()
                self._advance("release")
        elif self.phase == "release":
            if self.elapsed >= 0.35:
                if self.robot.holding(self.target):
                    raise RuntimeError("Gripper did not release product")
                low, high = self._product_bounds()
                surface = self.cell.placement(self.lane)
                roller_low, roller_high = self.cell.destination_bounds(self.lane)
                if (
                    abs(low[2] - surface[2]) > 0.065
                    or np.any(low[:2] < roller_low[:2] - 0.02)
                    or np.any(high[:2] > roller_high[:2] + 0.02)
                ):
                    raise RuntimeError(
                        "Product missed destination lane: "
                        + f"bounds={low.tolist()}..{high.tolist()}, "
                        + f"surface={surface.tolist()}"
                    )
                self.cell.mark_sorted(self.cell.product(self.target), self.lane)
                self._advance("retract")
        elif self.phase == "retract":
            if self._arrived(
                self.robot.move_pose(
                    self.place_above,
                    self.rotation,
                    dt,
                    target=self.target,
                    allowed_lane=self.lane,
                ),
                dt,
            ):
                return True
        if self.elapsed > 50:
            raise RuntimeError("Transport motion timed out")
        return False

    def finish(self):
        """Clear the completed transaction after returning Home."""
        self.target = self.phase = None
