"""Physical UR20 commands and the selected Lula motion planner."""

import time
from pathlib import Path

import numpy as np
import omni.physics.tensors
import omni.physx
import omni.usd
import warp as wp
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.robot.surface_gripper import _surface_gripper
from isaacsim.robot_motion.motion_generation import RmpFlow
from isaacsim.robot_motion.motion_generation.lula.path_planners import RRT
from pxr import Usd, UsdPhysics

from .geometry import JOINT_NAMES, ROBOT_PATH, rotation_error, transform
from .lula import TCP_FRAME, LulaModel, _quaternion
from .workcell import GRIPPER


class UR20Controller:
    """Own the physical arm, gripper, and one planner selected on reload."""

    def __init__(self, stage, workcell, planner_name, config):
        """Validate the arm and construct an obstacle-aware Lula planner."""
        if planner_name not in ("RMPflow", "RRT"):
            raise ValueError("PLANNER must be RMPflow or RRT")
        self.stage, self.workcell, self.planner_name = stage, workcell, planner_name
        self.config = config
        roots = [
            prim
            for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH))
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if len(roots) != 1:
            raise ValueError("UR20 must have one articulation root")
        self.root = str(roots[0].GetPath())
        self.stage_id = omni.usd.get_context().get_stage_id()
        self.model = LulaModel(stage, config.get("urdf"), config.get("description"))
        self.home = self.model.default_q.copy()
        self.gripper = _surface_gripper.acquire_surface_gripper_interface()
        self.simulation = self.view = None
        self.proxies = {}
        self.proxy_paths = {}
        self._next_proxy = 0
        self.obstacle_centers = {}
        self.obstacle_extents = {}
        self.planner = None
        self.path = None
        self.path_index = 0
        self.goal_key = None
        self._rrt_goal = None
        self._rrt_replans = 0
        self._rrt_started = 0.0
        self._obstacle_ticks = 0
        self._obstacle_context = None
        self._make_planner()

    def _make_planner(self):
        """Build only the selected Lula planner from validated model files."""
        key = "rmpflow" if self.planner_name == "RMPflow" else "rrt"
        bundled = Path(__file__).with_name(
            "rmpflow.yaml" if self.planner_name == "RMPflow" else "rrt.yaml"
        )
        path = Path(self.config.get(key) or bundled)
        if not path.is_file():
            if not self.config.get(key) or not bundled.is_file():
                raise ValueError("Planner configuration missing: " + str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundled.read_bytes())
        # Lula's native YAML loader cannot open non-ASCII Windows workspace paths.
        local_config = self.model.local_config_copy(path)
        settings = dict(
            robot_description_path=str(self.model.active_description_path),
            urdf_path=str(self.model.urdf_path),
            end_effector_frame_name=TCP_FRAME,
        )
        if self.planner_name == "RMPflow":
            self.planner = RmpFlow(
                **settings,
                rmpflow_config_path=str(local_config),
                maximum_substep_size=1 / 300,
            )
        else:
            self.planner = RRT(**settings, rrt_config_path=str(local_config))
        self.planner.set_robot_base_pose(
            self.model.base[:3, 3], _quaternion(self.model.base[:3, :3])
        )
        if self.planner.get_active_joints() != list(JOINT_NAMES):
            raise ValueError("Planner and UR20 joints differ")

    def set_payload(self, center=None, radius=None):
        """Rebuild the selected planner with the held carton collision sphere."""
        self.model.set_payload(center, radius)
        self._remove_proxies()
        self._obstacle_context = None
        self._obstacle_ticks = 0
        self._make_planner()
        self.goal_key = self.path = None

    def _remove_proxies(self):
        """Remove this controller's temporary obstacle Prims from the session."""
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for path in self.proxy_paths.values():
                self.stage.RemovePrim(path)
        self.proxies.clear()
        self.proxy_paths.clear()
        self.obstacle_centers.clear()
        self.obstacle_extents.clear()

    def _ensure_view(self):
        """Recreate the tensor articulation view after simulation resets."""
        if self.simulation is None or not self.simulation.is_valid:
            self.simulation = omni.physics.tensors.create_simulation_view(
                "warp", stage_id=self.stage_id
            )
            self.simulation.set_subspace_roots("/")
            self.view = self.simulation.create_articulation_view(self.root)
            if self.view.count != 1:
                raise RuntimeError("UR20 articulation unavailable")
            self.names = list(self.view.shared_metatype.dof_names)
            self.indices = [self.names.index(name) for name in JOINT_NAMES]

    def joints(self):
        """Read the six measured arm joint positions and velocities."""
        self._ensure_view()
        q = self.view.get_dof_positions().numpy()[0].copy()
        dq = self.view.get_dof_velocities().numpy()[0].copy()
        return q[self.indices], dq[self.indices]

    def command(self, goal):
        """Set six position references while preserving all other DOFs."""
        self._ensure_view()
        q = self.view.get_dof_positions().numpy()[0].copy()
        q[self.indices] = np.asarray(goal, dtype=float)
        self.view.set_dof_position_targets(
            wp.array(q[None, :], dtype=wp.float32, device=self.simulation.device),
            wp.array([0], dtype=wp.int32, device=self.simulation.device),
        )

    def initialize_home(self):
        """Align a fresh PhysX articulation with the authored USD Home pose."""
        self._ensure_view()
        q = self.view.get_dof_positions().numpy().copy()
        dq = self.view.get_dof_velocities().numpy().copy()
        if np.max(np.abs(q[0, self.indices] - self.home)) < np.deg2rad(0.5):
            return
        q[0, self.indices] = self.home
        dq[0, self.indices] = 0.0
        index = wp.array([0], dtype=wp.int32, device=self.simulation.device)
        # The referenced asset's initial rigid-body poses do not always match
        # its joint drives; set the physical articulation once before work.
        self.view.set_dof_positions(
            wp.array(q, dtype=wp.float32, device=self.simulation.device), index
        )
        self.view.set_dof_velocities(
            wp.array(dq, dtype=wp.float32, device=self.simulation.device), index
        )
        self.command(self.home)
        self.goal_key = self.path = None

    def hold(self):
        """Hold the measured pose and clear a planned trajectory."""
        q, _ = self.joints()
        self.command(q)
        self.goal_key = self.path = None

    def tcp(self):
        """Read the physical suction TCP pose, including body compliance."""
        value = omni.physx.get_physx_interface().get_rigidbody_transformation(
            ROBOT_PATH + "/long_gripper"
        )
        if not value.get("ret_val"):
            raise RuntimeError("Suction body unavailable")
        x, y, z, w = value["rotation"]
        return transform(value["position"], [w, x, y, z]) @ self.model.tip_frame

    def holding(self, product_path):
        """Check that the suction gripper actually attached the target product."""
        return any(
            str(path) == product_path or str(path).startswith(product_path + "/")
            for path in self.gripper.get_gripped_objects(GRIPPER)
        )

    def suction(self, closed):
        """Close or open the existing surface gripper."""
        if closed:
            self.gripper.close_gripper(GRIPPER)
        else:
            self.gripper.open_gripper(GRIPPER)

    def _route_blocked(self, low, high):
        """Check a changed box against every remaining RRT link sphere."""
        if self.path is None or self.goal_key is None:
            return False
        for joints in self.path[self.path_index :]:
            for center, radius in self.model.collision_spheres_at(joints):
                nearest = np.maximum(low, np.minimum(center, high))
                if np.linalg.norm(center - nearest) < radius + 0.01:
                    return True
        return False

    def _sync_obstacles(self, target=None, allowed_lane=None, allowed_contacts=()):
        """Update conservative planner boxes for active workcell colliders."""
        self._obstacle_ticks += 1
        context = (target, allowed_lane, tuple(allowed_contacts))
        if (
            context == self._obstacle_context
            and self._obstacle_ticks > 1
            and self._obstacle_ticks % 6
        ):
            return
        self._obstacle_context = context
        seen = set()
        changed = False
        world_changed = False
        for path, low, high in self.workcell.obstacles(
            target, allowed_lane, allowed_contacts
        ):
            if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
                raise RuntimeError("Invalid collision bounds: " + path)
            extent = np.maximum(high - low, 0.005)
            center = (low + high) * 0.5
            seen.add(path)
            previous = self.obstacle_centers.get(path)
            old_extent = self.obstacle_extents.get(path)
            moved = (
                previous is None
                or np.linalg.norm(center - previous) > 1e-4
                or np.linalg.norm(extent - old_extent) > 1e-4
            )
            if moved and self.planner_name == "RRT" and self._route_blocked(low, high):
                changed = True
            if path not in self.proxies:
                proxy_path = "/World/NGVPlannerProxy/box_" + str(self._next_proxy)
                self._next_proxy += 1
                with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                    cube = VisualCuboid(
                        proxy_path,
                        position=center,
                        scale=extent,
                        size=1.0,
                        visible=False,
                    )
                if not self.planner.add_obstacle(cube):
                    raise RuntimeError("Planner rejected obstacle " + path)
                self.proxies[path] = cube
                self.proxy_paths[path] = proxy_path
                world_changed = True
            else:
                if (
                    np.linalg.norm(center - previous) > 1e-4
                    or np.linalg.norm(extent - old_extent) > 1e-4
                ):
                    self.proxies[path].set_world_pose(position=center)
                    self.proxies[path].set_local_scale(extent)
                    world_changed = True
            self.obstacle_centers[path] = center
            self.obstacle_extents[path] = extent
        for path, cube in list(self.proxies.items()):
            if path not in seen:
                self.planner.remove_obstacle(cube)
                with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                    self.stage.RemovePrim(self.proxy_paths.pop(path))
                del self.proxies[path]
                self.obstacle_centers.pop(path, None)
                self.obstacle_extents.pop(path, None)
                world_changed = True
        if world_changed:
            self.planner.update_world()
        if changed and self.planner_name == "RRT":
            # Only a box crossing the remaining swept route invalidates it.
            self.goal_key = None

    def move_pose(
        self,
        position,
        rotation,
        dt,
        *,
        target=None,
        allowed_lane=None,
        allowed_contacts=(),
        tolerance=0.012,
    ):
        """Move the TCP with RMPflow or follow a fresh RRT route to a pose."""
        goal = np.asarray(position, dtype=float)
        self._sync_obstacles(target, allowed_lane, allowed_contacts)
        q, dq = self.joints()
        key = ("pose", *np.round(goal, 3), *np.round(rotation.flatten(), 3))
        if self.planner_name == "RMPflow":
            if key != self.goal_key:
                self.planner.set_cspace_target(
                    self.model.pose_joints(goal, rotation, q)
                )
                self.goal_key = key
            self.planner.set_end_effector_target(goal, _quaternion(rotation))
            command, _ = self.planner.compute_joint_targets(
                q, dq, np.array([]), np.array([]), min(dt * 4, 0.08)
            )
            # Advance the policy far enough for the workcell's high-mass
            # drives, while limiting each physical reference increment.
            self.command(q + np.clip(command - q, -0.03, 0.03))
        else:
            self._follow_rrt(key, q, goal, rotation)
        tcp = self.tcp()
        return (
            np.linalg.norm(goal - tcp[:3, 3]) < tolerance
            and np.linalg.norm(rotation_error(rotation, tcp[:3, :3])) < 0.08
        )

    def move_home(self, dt):
        """Return to the original USD Home using the selected planner."""
        q, dq = self.joints()
        if np.max(np.abs(q - self.home)) < np.deg2rad(0.5):
            self.command(self.home)
            return True
        self._sync_obstacles()
        if self.planner_name == "RMPflow":
            self.goal_key = ("home",)
            # Lula keeps the previous Cartesian attractor until explicitly cleared.
            self.planner.set_end_effector_target(None, None)
            self.planner.set_cspace_target(self.home)
            command, _ = self.planner.compute_joint_targets(
                q, dq, np.array([]), np.array([]), min(dt * 8, 0.15)
            )
            # The workcell's high-mass drive lags the small per-frame Lula
            # reference; cap each larger Home reference before applying it.
            self.command(q + np.clip(command - q, -0.04, 0.04))
        else:
            self._follow_rrt(("home",), q, None, None)
        return np.max(np.abs(q - self.home)) < np.deg2rad(0.5)

    def _follow_rrt(self, key, q, position, rotation):
        """Plan from measured joints once, then execute validated RRT samples."""
        if key != self._rrt_goal:
            self._rrt_goal, self._rrt_replans = key, 0
            self._rrt_started = 0.0
        if key != self.goal_key:
            if self.path is not None:
                if self._rrt_replans == 0:
                    self._rrt_started = time.monotonic()
                self._rrt_replans += 1
                if self._rrt_replans > 3 or time.monotonic() - self._rrt_started > 10:
                    raise RuntimeError("RRT replanning limit reached")
            if position is None:
                self.planner.set_cspace_target(self.home)
            else:
                self.planner.set_end_effector_target(position, _quaternion(rotation))
            self.path = self.planner.compute_path(q, np.array([]))
            if self.path is None and position is not None:
                # A task-space RRT can exhaust samples near a singular pose.
                # Solve only its terminal pose, then let the same RRT plan the
                # entire collision-aware route in joint space.
                terminal = self.model.pose_joints(position, rotation, q)
                self.planner.set_cspace_target(terminal)
                self.path = self.planner.compute_path(q, np.array([]))
            if self.path is None or len(self.path) == 0:
                raise RuntimeError("RRT found no safe route")
            self.path = np.asarray(self.path, dtype=float)
            if (
                self.path.ndim != 2
                or self.path.shape[1] != len(JOINT_NAMES)
                or not np.isfinite(self.path).all()
            ):
                raise RuntimeError("RRT returned an invalid route")
            # Revolute samples may cross the +/-pi representation boundary.
            # Keep equivalent poses on the branch nearest the measured arm.
            if position is not None:
                previous = q.copy()
                for waypoint in self.path:
                    for joint, (lower, upper) in enumerate(self.model.limits):
                        candidates = [
                            waypoint[joint] + 2 * np.pi * turn
                            for turn in (-1, 0, 1)
                            if lower <= waypoint[joint] + 2 * np.pi * turn <= upper
                        ]
                        if not candidates:
                            raise RuntimeError("RRT route exceeds a UR20 joint limit")
                        waypoint[joint] = min(
                            candidates, key=lambda value: abs(value - previous[joint])
                        )
                    previous = waypoint
            self.path_index, self.goal_key = 0, key
        waypoint = self.path[self.path_index]
        self.command(waypoint)
        if (
            np.max(np.abs(q - waypoint)) < 0.035
            and self.path_index < len(self.path) - 1
        ):
            self.path_index += 1

    def close(self):
        """Release tensor and temporary model resources."""
        self.view = self.simulation = None
        self._remove_proxies()
        self.model.close()
