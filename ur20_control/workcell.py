"""USD and PhysX facts used by the independent UR20 behavior."""

import numpy as np
import omni.physx
import omni.timeline
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from .geometry import ATTACHMENT_PATH, JOINT_NAMES, authored_rigid_pose, transform

OWNER = "ngv.product_loader"
ROBOT = "/World/Robot/ur20_gripper/ur20"
GRIPPER = ROBOT + "/long_gripper/SurfaceGripper"
GUARD = "/World/Lane01/Guard"
PICKUP_BELT = "/World/Lane01/ConveyorBelt_A02"
ROUTES = {
    "Product_1": ("Lane02", "/World/Lane02/ConveyorBelt_A05_03/Rollers"),
    "Product_2": ("Lane03", "/World/Lane03/ConveyorBelt_A05_03/Rollers"),
}
INFEED = tuple(
    "/World/Graphs/Lane01ActionGraph/" + name
    for name in ("conveyor_belt", "conveyor_belt_01", "conveyor_belt_02")
)
PRODUCTS_PATH = Sdf.Path("/World/NGVProducts")


def owned_product(stage, collider_path):
    """Resolve a collider path to an active loader product."""
    path = Sdf.Path(str(collider_path))
    if not path.IsAbsolutePath() or not path.HasPrefix(PRODUCTS_PATH):
        return None
    while path != PRODUCTS_PATH:
        prim = stage.GetPrimAtPath(path)
        if (
            prim
            and prim.IsActive()
            and prim.GetCustomDataByKey("ngv:owner") == OWNER
            and prim.GetCustomDataByKey("ngv:product") in ROUTES
        ):
            return prim
        path = path.GetParentPath()
    return None


def bounds(prim):
    """Return conservative world bounds for a scene Prim."""
    box = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    if box.IsEmpty():
        raise ValueError("No bounds for " + str(prim.GetPath()))
    return np.asarray(box.GetMin(), dtype=float), np.asarray(box.GetMax(), dtype=float)


def physical_bounds(prim, box, poses):
    """Move authored collider bounds with their measured PhysX rigid body."""
    low = np.asarray(box.GetMin(), dtype=float)
    high = np.asarray(box.GetMax(), dtype=float)
    body = prim
    while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
        body = body.GetParent()
    if (
        not body
        or UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() is True
        or not omni.timeline.get_timeline_interface().is_playing()
    ):
        return low, high
    path = str(body.GetPath())
    if path not in poses:
        value = omni.physx.get_physx_interface().get_rigidbody_transformation(path)
        if not value.get("ret_val"):
            poses[path] = None
        else:
            x, y, z, w = value["rotation"]
            measured = transform(value["position"], [w, x, y, z])
            authored = authored_rigid_pose(body)
            poses[path] = measured @ np.linalg.inv(authored)
    delta = poses[path]
    if delta is None:
        return low, high
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )
    world = corners @ delta.T
    return world[:, :3].min(axis=0), world[:, :3].max(axis=0)


class Workcell:
    """Keep scene queries and reversible conveyor/metadata edits together."""

    def __init__(self, stage):
        """Validate the known workcell and record the original input speeds."""
        self.stage = stage
        self.speeds = {}
        for path in (ROBOT, GUARD, *(route[1] for route in ROUTES.values()), *INFEED):
            if not stage.GetPrimAtPath(path):
                raise ValueError("Missing workcell Prim: " + path)
        for path in INFEED:
            attr = stage.GetPrimAtPath(path).GetAttribute("inputs:velocity")
            speed = next(
                (
                    spec.default
                    for spec in attr.GetPropertyStack()
                    if spec.layer.identifier != stage.GetSessionLayer().identifier
                    and spec.HasDefaultValue()
                ),
                None,
            )
            if speed is None:
                raise ValueError("Missing conveyor velocity at " + path)
            self.speeds[path] = float(speed)
        self.guard_bounds = bounds(stage.GetPrimAtPath(GUARD))
        self.base_position = np.asarray(
            UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT + "/base_link"))
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .ExtractTranslation()
        )
        self.query = omni.physx.get_physx_scene_query_interface()
        self.configure_robot_physics()

    def configure_robot_physics(self):
        """Apply the existing suction and drive requirements in session USD."""
        attachment = self.stage.GetPrimAtPath(ATTACHMENT_PATH)
        if not attachment:
            raise ValueError("Missing suction attachment joint")
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            # These D6 settings make the attachment behave as a fixed suction
            # connection while keeping the referenced robot USD untouched.
            attachment.AddAppliedSchema("IsaacAttachmentPointAPI")
            for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
                UsdPhysics.LimitAPI.Apply(attachment, axis).CreateLowAttr(1.0)
                UsdPhysics.LimitAPI(attachment, axis).CreateHighAttr(-1.0)
                if attachment.HasAPI(UsdPhysics.DriveAPI, axis):
                    UsdPhysics.DriveAPI.Apply(
                        attachment, axis
                    ).CreateTargetPositionAttr(0.0)
            UsdPhysics.Joint(attachment).CreateExcludeFromArticulationAttr(True)
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(ROBOT)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.AddAppliedSchema("PhysxRigidBodyAPI")
                    prim.CreateAttribute(
                        "physxRigidBody:disableGravity",
                        Sdf.ValueTypeNames.Bool,
                        custom=False,
                    ).Set(True)
            for name in JOINT_NAMES:
                joint = self.stage.GetPrimAtPath(ROBOT + "/joints/" + name)
                angle = (
                    joint.GetAttribute("drive:angular:physics:targetPosition").Get()
                    or 0.0
                )
                # PhysX ignores bare state:* attributes without this API.
                state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
                state.CreatePositionAttr(angle)
                state.CreateVelocityAttr(0.0)

    def brake(self, stopped):
        """Set all Lane01 drives in the session layer without changing the USD asset."""
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for path, speed in self.speeds.items():
                self.stage.GetPrimAtPath(path).GetAttribute("inputs:velocity").Set(
                    0.0 if stopped else speed
                )

    def interrupted(self):
        """Report whether a previous script reload interrupted an active task."""
        return bool(self.stage.GetPrimAtPath(ROBOT).GetCustomDataByKey("ngv:interrupted"))

    def mark_interrupted(self, interrupted):
        """Retain a safe-stop marker across Behavior Script reloads."""
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.stage.GetPrimAtPath(ROBOT).SetCustomDataByKey(
                "ngv:interrupted", bool(interrupted)
            )

    def product(self, path):
        """Return an active product or reject stale and foreign event paths."""
        return owned_product(self.stage, path)

    def placement(self, lane, half_extents=None):
        """Fit the measured product footprint near the reachable belt edge."""
        low, high = self.destination_bounds(lane)
        base = (
            UsdGeom.Xformable(self.stage.GetPrimAtPath(ROBOT + "/base_link"))
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .ExtractTranslation()
        )
        # Keep the existing reachable preference, then constrain it by the
        # actual footprint instead of assuming every carton is the same size.
        x = high[0] - 0.20 if base[0] > (low[0] + high[0]) * 0.5 else low[0] + 0.20
        preferred = np.array([x, high[1] - 0.22])
        if half_extents is not None:
            half_extents = np.asarray(half_extents, dtype=float)
            if (
                half_extents.shape != (2,)
                or not np.isfinite(half_extents).all()
                or np.any(half_extents <= 0)
            ):
                raise ValueError("Invalid product footprint for " + lane)
            minimum = low[:2] + half_extents + 0.02
            maximum = high[:2] - half_extents - 0.02
            if np.any(minimum > maximum):
                raise RuntimeError(
                    "Product does not fit destination " + lane
                    + f": footprint={(2 * half_extents).tolist()}, "
                    + f"available={(high[:2] - low[:2] - 0.04).tolist()}"
                )
            preferred = np.clip(preferred, minimum, maximum)
        return np.r_[preferred, high[2]]

    def destination_bounds(self, lane):
        """Return the real roller surface bounds for placement verification."""
        roller = next(path for route_lane, path in ROUTES.values() if route_lane == lane)
        return bounds(self.stage.GetPrimAtPath(roller))

    def central_grasp(self, product, low, high, xy=None):
        """Verify five top-facing ray hits around the product center."""
        if xy is None:
            xy = (low[:2] + high[:2]) * 0.5
        hits = []
        for dx, dy in ((0, 0), (0.009, 0), (-0.009, 0), (0, 0.009), (0, -0.009)):
            hit = self.query.raycast_closest(
                Gf.Vec3f(float(xy[0] + dx), float(xy[1] + dy), float(high[2] + 0.15)),
                Gf.Vec3f(0, 0, -1),
                0.5,
                bothSides=True,
            )
            if not hit.get("hit") or not any(
                (owner := owned_product(self.stage, hit.get(key, "")))
                and owner.GetPath() == product.GetPath()
                for key in ("rigidBody", "collision")
            ):
                raise RuntimeError(
                    "No supported central suction patch on " + str(product.GetPath())
                )
            hits.append(np.asarray(hit["position"], dtype=float))
        if np.ptp(np.asarray(hits)[:, 2]) > 0.012:
            raise RuntimeError("Uneven suction patch on " + str(product.GetPath()))
        return hits[0] + np.array([0, 0, 0.009])

    def support_collider(self, product, low, high):
        """Identify the collider directly supporting a stopped product."""
        candidates = []
        for path, floor, ceiling in self.obstacles(str(product.GetPath())):
            if (
                np.all(ceiling[:2] > low[:2])
                and np.all(floor[:2] < high[:2])
                and low[2] - 0.10 <= ceiling[2] <= low[2] + 0.03
            ):
                candidates.append((abs(ceiling[2] - low[2]), path))
        if not candidates:
            raise RuntimeError("No collider supports the product at the Guard")
        return min(candidates)[1]

    def mark_sorted(self, product, lane):
        """Record verified placement in the session layer for the output tunnel."""
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            product.SetCustomDataByKey("ngv:sortedLane", lane)

    def obstacles(self, target=None, allowed_lane=None, allowed_contacts=()):
        """Enumerate active collision-bearing bodies outside the robot and target."""
        ignored = (Sdf.Path(ROBOT), Sdf.Path(target) if target else None)
        contacts = tuple(Sdf.Path(contact) for contact in allowed_contacts)
        result = []
        selected = []
        poses = {}
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), ["default", "render", "proxy"]
        )
        for prim in self.stage.Traverse():
            path = prim.GetPath()
            if any(item and path.HasPrefix(item) for item in ignored):
                continue
            if any(
                path.HasPrefix(contact) or contact.HasPrefix(path)
                for contact in contacts
            ):
                continue
            if any(path.HasPrefix(parent) for parent in selected):
                continue
            if allowed_lane and path.HasPrefix(
                Sdf.Path("/World/" + allowed_lane + "/ConveyorBelt_A05_03")
            ):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if "PhysxTriggerAPI" in prim.GetAppliedSchemas():
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            # Each collider contributes a conservative box; complex meshes may
            # enlarge the route, and physical contacts are checked separately.
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            low, high = physical_bounds(prim, box, poses)
            nearest = np.maximum(low, np.minimum(self.base_position, high))
            if np.linalg.norm(nearest - self.base_position) <= 2.5:
                result.append((str(path), low, high))
                selected.append(path)
        return result
