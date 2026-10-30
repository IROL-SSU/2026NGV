"""Fail-closed physical contact monitoring for the UR20 workcell."""

from collections import deque

import omni.physx
from omni.physx.bindings import _physx
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdPhysics

from .workcell import GUARD, PICKUP_BELT, ROBOT, owned_product


class SafetyMonitor:
    """Detect robot or held-product contacts outside intentional task pairs."""

    def __init__(self, stage, stage_id):
        """Enable contact reports on robot collision bodies in the session."""
        self.stage, self.stage_id = stage, stage_id
        self.pending = deque()
        self.subscription = None
        self.reported_products = set()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(
                        0.0
                    )
        group = stage.GetPrimAtPath("/World/NGVProducts")
        if group:
            for product in group.GetChildren():
                if owned_product(stage, product.GetPath()):
                    self.watch_product(str(product.GetPath()))

    def watch_product(self, path):
        """Enable zero-threshold contact reporting on each product collider."""
        if path in self.reported_products:
            return
        product = owned_product(self.stage, path)
        if not product:
            return
        found = False
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for prim in Usd.PrimRange(product):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(
                        0.0
                    )
                    found = True
        if found:
            self.reported_products.add(path)

    def play(self):
        """Subscribe to PhysX contact reports on simulation Play."""
        if self.subscription is None:
            self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                self._contact
            )

    def _contact(self, headers, data):
        """Queue contact pairs without editing the scene in a physics callback."""
        for header in headers:
            if (
                header.stage_id != self.stage_id
                or header.type == _physx.ContactEventType.CONTACT_LOST
            ):
                continue
            self.pending.append(
                (
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
            )

    def check(self, machine):
        """Reject unexpected contacts involving the arm or its held product."""
        if machine.tasks.target:
            self.watch_product(machine.tasks.target)
        while self.pending:
            left, right = self.pending.popleft()
            target = machine.tasks.target
            held = bool(target and machine.robot.holding(target))
            robot_left, robot_right = (
                left.startswith(ROBOT + "/"),
                right.startswith(ROBOT + "/"),
            )
            product_left = bool(held and left.startswith(target))
            product_right = bool(held and right.startswith(target))
            if not any((robot_left, robot_right, product_left, product_right)):
                continue
            if robot_left and robot_right:
                continue
            # Intentional contact is limited to suction/target, product/Guard,
            # and product/destination lane during verified release.
            pair = {left, right}
            if target and any(path.startswith(target) for path in pair):
                other = right if left.startswith(target) else left
                if other.startswith(ROBOT + "/long_gripper/suction_cup/"):
                    continue
                if other.startswith(GUARD) and machine.state == "Picking":
                    continue
                if (
                    other.startswith(PICKUP_BELT)
                    and machine.state == "Picking"
                    and machine.tasks.phase in ("attach", "clear_guard", "lift")
                ):
                    continue
                support = getattr(machine.tasks, "support_path", "")
                if (
                    support
                    and machine.state == "Picking"
                    and machine.tasks.phase in ("attach", "clear_guard", "lift")
                    and (other.startswith(support) or support.startswith(other))
                ):
                    continue
                if machine.state == "Carrying" and machine.tasks.phase in (
                    "lower",
                    "release",
                ):
                    lane = "/World/" + machine.tasks.lane + "/"
                    if other.startswith(lane):
                        continue
            raise RuntimeError(
                "Unexpected physical contact: " + left + " with " + right
            )

    def stop(self):
        """Release the contact subscription and discard stale callbacks."""
        self.subscription = None
        self.pending.clear()
