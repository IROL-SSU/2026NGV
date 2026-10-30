"""Lula FK/IK for the inspected USD arm, including its actual suction frame.

The URDF and Lula description are generated in a private temporary directory.
They contain kinematics only; the Stage remains the source of physics/geometry.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .geometry import JOINT_NAMES, ROBOT_PATH, USDJointGeometry, rotation_error

TCP_FRAME = "ngv_suction_tcp"


def _quaternion(matrix):
    """A column-vector rotation matrix to Lula's scalar-first quaternion."""
    value = Gf.Matrix3d(*np.asarray(matrix).T.reshape(-1)).ExtractRotation().GetQuat()
    return np.array([value.GetReal(), *value.GetImaginary()], dtype=float)


# Encode a USD joint transform as URDF xyz and fixed-axis roll/pitch/yaw
def _origin(joint, frame):
    """Write a USD joint frame as a URDF origin transform."""
    r = frame[:3, :3]
    horizontal = np.hypot(r[0, 0], r[1, 0])
    pitch = np.arctan2(-r[2, 0], horizontal)
    if horizontal > 1e-9:
        roll, yaw = np.arctan2(r[2, 1], r[2, 2]), np.arctan2(r[1, 0], r[0, 0])
    else:
        roll, yaw = np.arctan2(-r[1, 2], r[1, 1]), 0.0
    values = lambda items: " ".join(format(float(v), ".17g") for v in items)
    ET.SubElement(
        joint, "origin", xyz=values(frame[:3, 3]), rpy=values([roll, pitch, yaw])
    )


def _write_model(model, default_q, directory, spheres):
    """Represent both USD local joint frames, rather than assuming DH offsets."""
    robot = ET.Element("robot", name="ngv_ur20_suction")
    ET.SubElement(robot, "link", name="base_link")
    parent = "base_link"
    for index, (name, frames, limits) in enumerate(
        zip(JOINT_NAMES, model.frames, model.limits)
    ):
        revolute_frame, child = f"joint_frame_{index}", f"arm_link_{index}"
        ET.SubElement(robot, "link", name=revolute_frame)
        joint = ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=revolute_frame)
        _origin(joint, frames[0])
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(
            joint,
            "limit",
            lower=str(limits[0]),
            upper=str(limits[1]),
            effort="1000",
            velocity="3.14",
        )
        ET.SubElement(robot, "link", name=child)
        fixed = ET.SubElement(robot, "joint", name=f"child_frame_{index}", type="fixed")
        ET.SubElement(fixed, "parent", link=revolute_frame)
        ET.SubElement(fixed, "child", link=child)
        _origin(fixed, frames[1])
        parent = child
    ET.SubElement(robot, "link", name=TCP_FRAME)
    tip = ET.SubElement(robot, "joint", name="ngv_tool_joint", type="fixed")
    ET.SubElement(tip, "parent", link=parent)
    ET.SubElement(tip, "child", link=TCP_FRAME)
    _origin(tip, model.tool)
    urdf = Path(directory) / "ur20_suction.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf, encoding="utf-8", xml_declaration=True)
    description = Path(directory) / "ur20_robot_description.yaml"
    # JSON is a YAML subset and avoids an additional YAML-writing dependency.
    description.write_text(
        json.dumps(
            {
                "api_version": 1.0,
                "cspace": list(JOINT_NAMES),
                "root_link": "base_link",
                "subtree_root_link": "base_link",
                "default_q": np.asarray(default_q).tolist(),
                "cspace_to_urdf_rules": [],
                "composite_task_spaces": [],
                "collision_spheres": spheres,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return urdf, description


def _fit_spheres(low, high, frame):
    """Cover a world-space box with spheres in one robot link frame."""
    corners = np.array(
        [
            [x, y, z, 1]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )
    local = corners @ np.linalg.inv(frame).T
    low, high = local[:, :3].min(axis=0), local[:, :3].max(axis=0)
    sizes = high - low
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("Invalid robot collision bounds")
    axis = int(np.argmax(sizes))
    other = [size for number, size in enumerate(sizes) if number != axis]
    count = max(1, min(12, int(np.ceil(sizes[axis] / max(min(other), 0.06)))))
    segment = sizes[axis] / count
    radius = float(np.linalg.norm([segment, *other]) * 0.5)
    result = []
    for sample in range(count):
        center = (low + high) * 0.5
        center[axis] = low[axis] + (sample + 0.5) * segment
        result.append({"center": center.tolist(), "radius": radius})
    return result


def _fit_tool_spheres(low, high, frame):
    """Cover the narrow gripper housing without inflating its Guard clearance."""
    corners = np.array(
        [
            [x, y, z, 1]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )
    local = corners @ np.linalg.inv(frame).T
    lower, upper = local[:, :3].min(axis=0), local[:, :3].max(axis=0)
    sizes = upper - lower
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("Invalid gripper collision bounds")
    counts = np.maximum(1, np.ceil(sizes / 0.08).astype(int))
    cell = sizes / counts
    radius = float(np.linalg.norm(cell) * 0.5)
    return [
        {"center": (lower + (np.array(index) + 0.5) * cell).tolist(), "radius": radius}
        for index in np.ndindex(*counts)
    ]


def _validate_spheres(supplied, expected):
    """Require supplied spheres to enclose the inspected robot geometry."""
    if not isinstance(supplied, list):
        raise ValueError("Robot collision spheres must be a list")
    groups = {}
    for item in supplied:
        if not isinstance(item, dict):
            raise ValueError("Invalid robot collision sphere group")
        for link, spheres in item.items():
            if not isinstance(spheres, list):
                raise ValueError("Invalid collision spheres for " + link)
            groups.setdefault(link, []).extend(spheres)
    for expected_group in expected:
        for link, spheres in expected_group.items():
            actual = groups.get(link, [])
            for sphere in spheres:
                center = np.asarray(sphere["center"], dtype=float)
                radius = float(sphere["radius"])
                covered = False
                for other in actual:
                    point = np.asarray(other.get("center"), dtype=float)
                    outer = float(other.get("radius", -1))
                    if (
                        point.shape == (3,)
                        and np.isfinite(point).all()
                        and np.isfinite(outer)
                        and outer > 0
                        and np.linalg.norm(center - point) + radius <= outer + 1e-5
                    ):
                        covered = True
                        break
                if not covered:
                    raise ValueError("Collision spheres do not cover " + link)


class LulaModel:
    """Own the validated UR20 kinematic model and generated configuration files."""

    backend = "LulaKinematicsSolver"

    def __init__(self, stage, urdf_path=None, description_path=None):
        """Validate or generate a Lula model for this robot's USD frames."""
        if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-8:
            raise ValueError("The Lula sorting model requires a Stage in meters.")
        # Geometry reads authored frames; Lula performs all actual IK/planning.
        geometry = USDJointGeometry(stage)
        self.base, self.tip_frame, self.limits = (
            geometry.base,
            geometry.tip_frame,
            geometry.limits,
        )
        self.default_q = np.deg2rad(
            [
                stage.GetPrimAtPath(ROBOT_PATH + "/joints/" + name)
                .GetAttribute("drive:angular:physics:targetPosition")
                .Get()
                or 0.0
                for name in JOINT_NAMES
            ]
        )
        self._directory = TemporaryDirectory(prefix="ngv_ur20_lula_")
        try:
            if bool(urdf_path) != bool(description_path):
                raise ValueError("Provide both URDF and robot description, or neither.")
            spheres = self._collision_spheres(stage)
            generated_urdf, generated_description = _write_model(
                geometry, self.default_q, self._directory.name, spheres
            )
            if urdf_path:
                self.urdf_path, self.description_path = (
                    Path(urdf_path),
                    Path(description_path),
                )
                for source, target in (
                    (generated_urdf, self.urdf_path),
                    (generated_description, self.description_path),
                ):
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(source.read_bytes())
                description = yaml.safe_load(
                    self.description_path.read_text(encoding="utf-8")
                )
                if not description or not description.get("collision_spheres"):
                    raise ValueError(
                        "The supplied Lula model has no collision spheres."
                    )
                _validate_spheres(description["collision_spheres"], spheres)
                # Native Lula resolves paths through a narrow Windows API.
                # Keep the source files untouched and load ASCII temp copies.
                source_urdf, source_description = self.urdf_path, self.description_path
                self.urdf_path = Path(self._directory.name) / "supplied_robot.urdf"
                self.description_path = (
                    Path(self._directory.name) / "supplied_description.yaml"
                )
                self.urdf_path.write_bytes(source_urdf.read_bytes())
                self.description_path.write_bytes(source_description.read_bytes())
            else:
                # The key changes with authored joints, tool, or collision mesh.
                # Keep a reusable source pair while loading ASCII temp copies.
                digest = hashlib.sha256(
                    generated_urdf.read_bytes() + generated_description.read_bytes()
                ).hexdigest()[:16]
                cache = Path.home() / ".ngv_product_loader" / "models" / digest
                cache.mkdir(parents=True, exist_ok=True)
                for source in (generated_urdf, generated_description):
                    saved = cache / source.name
                    if saved.exists():
                        if saved.read_bytes() != source.read_bytes():
                            raise ValueError("Cached Lula model changed: " + str(saved))
                    else:
                        saved.write_bytes(source.read_bytes())
                self.urdf_path, self.description_path = (
                    generated_urdf,
                    generated_description,
                )
            self.collision_spheres = yaml.safe_load(
                self.description_path.read_text(encoding="utf-8")
            )["collision_spheres"]
            self.active_description_path = self.description_path
            self.solver = LulaKinematicsSolver(
                str(self.description_path), str(self.urdf_path)
            )
            if self.solver.get_joint_names() != list(JOINT_NAMES):
                raise ValueError("Lula and PhysX arm joint orders do not match.")
            self.solver.set_robot_base_pose(
                self.base[:3, 3], _quaternion(self.base[:3, :3])
            )
            self.solver.sampling_seed = 7
            self.solver.max_num_descents = 1
            self._ik_cache = {}
            for q in (
                self.default_q,
                self.default_q + np.array([0.1, 0.1, -0.1, 0.1, 0.1, 0.1]),
            ):
                np.testing.assert_allclose(
                    self.forward(q),
                    geometry.forward(q),
                    atol=2e-6,
                    err_msg="The generated Lula TCP does not match the USD arm.",
                )
        except BaseException:
            self.close()
            raise

    def set_payload(self, center=None, radius=None):
        """Add or remove a conservative held-product sphere at the suction TCP."""
        source = yaml.safe_load(self.description_path.read_text(encoding="utf-8"))
        if center is not None:
            center = np.asarray(center, dtype=float)
            if (
                center.shape != (3,)
                or not np.isfinite(center).all()
                or not np.isfinite(radius)
                or radius <= 0
            ):
                raise ValueError("Invalid held-product collision geometry")
            tool = next(
                (item for item in source["collision_spheres"] if TCP_FRAME in item),
                None,
            )
            if tool is None:
                tool = {TCP_FRAME: []}
                source["collision_spheres"].append(tool)
            tool[TCP_FRAME].append({"center": center.tolist(), "radius": float(radius)})
        target = Path(self._directory.name) / "active_robot_description.yaml"
        target.write_text(yaml.safe_dump(source), encoding="utf-8")
        self.active_description_path = target
        self.collision_spheres = source["collision_spheres"]

    def local_config_copy(self, source):
        """Copy a planner config into Lula's ASCII-safe temporary directory."""
        target = Path(self._directory.name) / Path(source).name
        target.write_bytes(Path(source).read_bytes())
        return target

    def collision_spheres_at(self, joints, frame=None):
        """Yield world centers and radii of modeled link collision spheres."""
        for group in self.collision_spheres:
            for link, spheres in group.items():
                if frame is not None and link != frame:
                    continue
                origin, rotation = self.solver.compute_forward_kinematics(
                    link, joints
                )
                for sphere in spheres:
                    center = np.asarray(origin) + rotation @ np.asarray(
                        sphere["center"]
                    )
                    yield center, float(sphere["radius"])

    def pose_joints(self, position, rotation, seed):
        """Solve one terminal pose to guide RMPflow without generating a path."""
        position = np.asarray(position, dtype=float)
        key = (*np.round(position, 4), *np.round(rotation.flatten(), 4))
        if key in self._ik_cache:
            return self._ik_cache[key].copy()
        for warm_start in (np.asarray(seed, dtype=float), self.default_q):
            joints, success = self.solver.compute_inverse_kinematics(
                TCP_FRAME,
                position,
                _quaternion(rotation),
                warm_start=warm_start,
                position_tolerance=0.003,
                orientation_tolerance=0.03,
            )
            joints = np.asarray(joints, dtype=float)
            if (
                not success
                or joints.shape != (len(JOINT_NAMES),)
                or not np.isfinite(joints).all()
            ):
                continue
            actual = self.forward(joints)
            if (
                np.linalg.norm(actual[:3, 3] - position) <= 0.005
                and np.linalg.norm(rotation_error(rotation, actual[:3, :3])) <= 0.05
            ):
                self._ik_cache[key] = joints.copy()
                return joints
        raise RuntimeError("Lula cannot solve the required suction target")

    def forward(self, joints):
        """Compute the world suction pose using Lula forward kinematics."""
        position, orientation = self.solver.compute_forward_kinematics(
            TCP_FRAME, np.asarray(joints, dtype=float)
        )
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = orientation, np.asarray(position).reshape(3)
        return result

    def _collision_spheres(self, stage):
        """Conservatively fit a sphere to each collision-bearing UR20 link."""
        # Collision meshes are authored in the body's USD pose. That pose can
        # differ from Home, so never convert their bounds through Home FK.
        spheres = []
        for index, name in enumerate(JOINT_NAMES):
            joint = stage.GetPrimAtPath(ROBOT_PATH + "/joints/" + name)
            targets = UsdPhysics.Joint(joint).GetBody1Rel().GetTargets()
            if len(targets) != 1:
                raise ValueError("Cannot identify the collision body for " + name)
            body = stage.GetPrimAtPath(targets[0])
            frame = np.asarray(
                UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
            ).T
            collision_root = body.GetChild("collisions")
            if not collision_root:
                raise ValueError("No designated collision geometry for " + name)
            # The robot's collision meshes are instance proxies, so normal
            # PrimRange traversal would miss almost the entire arm.
            colliders = [
                prim
                for prim in Usd.PrimRange(collision_root, Usd.TraverseInstanceProxies())
                if UsdGeom.Mesh(prim)
            ]
            vertices = []
            for mesh in colliders:
                points = np.asarray(
                    UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=float
                )
                if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                    continue
                # BBoxCache reports empty boxes for these instance proxies.
                matrix = np.asarray(
                    UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                )
                vertices.append(np.c_[points, np.ones(len(points))] @ matrix)
            if not vertices:
                raise ValueError("No collision geometry for UR20 link " + name)
            world_points = np.concatenate(vertices)[:, :3]
            low, high = world_points.min(axis=0), world_points.max(axis=0)
            spheres.append({f"arm_link_{index}": _fit_spheres(low, high, frame)})
        if len(spheres) != len(JOINT_NAMES):
            raise ValueError("Incomplete UR20 collision model")
        gripper = stage.GetPrimAtPath(ROBOT_PATH + "/long_gripper")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), ["default", "proxy", "render"]
        )
        boxes = [
            cache.ComputeWorldBound(prim).ComputeAlignedRange()
            for prim in Usd.PrimRange(gripper, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        boxes = [box for box in boxes if not box.IsEmpty()]
        if not boxes:
            raise ValueError("No suction housing collision geometry")
        tool_low = np.min([np.asarray(box.GetMin()) for box in boxes], axis=0)
        tool_high = np.max([np.asarray(box.GetMax()) for box in boxes], axis=0)
        tool_frame = (
            np.asarray(
                UsdGeom.Xformable(gripper).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
            ).T
            @ self.tip_frame
        )
        spheres.append({TCP_FRAME: _fit_tool_spheres(tool_low, tool_high, tool_frame)})
        return spheres

    def close(self):
        """Remove temporary model files owned by this instance."""
        if getattr(self, "_directory", None) is not None:
            self._directory.cleanup()
            self._directory = None
