"""UR20 kinematics from its actual USD joint frames, with no UR10 model assumptions."""

import numpy as np
from pxr import Gf, Usd, UsdGeom

ROBOT_PATH = "/World/Robot/ur20_gripper/ur20"
GRIPPER_BODY = ROBOT_PATH + "/long_gripper"
ATTACHMENT_PATH = GRIPPER_BODY + "/suction_cup/suction_joint"
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def rotation(quat):
    """Convert a scalar-first quaternion to a rotation matrix."""
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform(position, quat):
    """Build a homogeneous transform from a position and quaternion."""
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation(quat), position
    return result


def authored_rigid_pose(prim):
    """Read the USD position/rotation frame used to retain scaled geometry."""
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    # PhysX reports a rigid pose, while USD may also contain root/ancestor scale.
    # Removing only this rigid frame leaves that scale in the body geometry.
    quat = Gf.Transform(matrix).GetRotation().GetQuat()
    return transform(
        matrix.ExtractTranslation(), [quat.GetReal(), *quat.GetImaginary()]
    )


def usd_frame(prim, suffix):
    """Read one local USD physics joint frame."""
    p = prim.GetAttribute("physics:localPos" + suffix).Get() or Gf.Vec3f(0)
    q = prim.GetAttribute("physics:localRot" + suffix).Get() or Gf.Quatf(1)
    return transform(p, [q.GetReal(), *q.GetImaginary()])


def rotation_error(goal, current):
    """World-frame rotation vector, including rotations close to 180 degrees."""
    relative = goal @ current.T
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    if np.trace(relative) > 2.9999:
        # Gf's axis-angle extraction loses precision for sub-milliradian angles.
        return skew * 0.5
    q = Gf.Matrix3d(*relative.T.reshape(-1)).ExtractRotation().GetQuat()
    w, v = float(q.GetReal()), np.array(q.GetImaginary())
    if w < 0:
        w, v = -w, -v
    length = np.linalg.norm(v)
    return 2 * v if length < 1e-9 else v * (2 * np.arctan2(length, w) / length)


class USDJointGeometry:
    """Read the actual USD joint frames for model generation and validation."""

    def __init__(self, stage):
        """Read six joint frames, limits, and the suction TCP from the scene."""
        self.base = np.array(
            UsdGeom.Xformable(
                stage.GetPrimAtPath(ROBOT_PATH + "/base_link")
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ).T
        self.frames, self.limits = [], []
        for name in JOINT_NAMES:
            joint = stage.GetPrimAtPath(ROBOT_PATH + "/joints/" + name)
            if not joint or joint.GetAttribute("physics:axis").Get() != "Z":
                raise ValueError("The UR20 joint chain has changed: " + name)
            self.frames.append(
                (usd_frame(joint, "0"), np.linalg.inv(usd_frame(joint, "1")))
            )
            self.limits.append(
                np.deg2rad(
                    [
                        joint.GetAttribute("physics:lowerLimit").Get(),
                        joint.GetAttribute("physics:upperLimit").Get(),
                    ]
                )
            )
        fixed = stage.GetPrimAtPath(GRIPPER_BODY + "/FixedJoint")
        self.tip_frame = usd_frame(stage.GetPrimAtPath(ATTACHMENT_PATH), "0")
        self.tool = (
            usd_frame(fixed, "0")
            @ np.linalg.inv(usd_frame(fixed, "1"))
            @ self.tip_frame
        )
        self.limits = np.array(self.limits)

    def forward(self, joints):
        """Compute the suction TCP transform for model consistency checks."""
        result = self.base.copy()
        for angle, (parent, child) in zip(joints, self.frames):
            result = result @ parent
            c, s = np.cos(angle), np.sin(angle)
            turn = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            result = result @ turn @ child
        return result @ self.tool
