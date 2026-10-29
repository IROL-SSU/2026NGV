
"""GuardContact 이벤트를 수신하여 Prim을 Delta만큼 이동시키는 BehaviorScript."""

import carb
import carb.events
import omni.kit.app

from omni.behavior.scripting.core import BehaviorScript
from pxr import Gf, UsdGeom


EVENT_NAME = "ngv.tutorial.guard_contact"

# Local Frame 기준 이동량 (Stage 단위, 일반적으로 meter)
DELTA_POSITION = (2.0, -1.5, 0.1)


class GuardEventMover(BehaviorScript):
    def on_init(self):
        self._subscription = None
        self._pending = False
        self._moved = False
        self._initial_position = None

        self._bus = omni.kit.app.get_app().get_message_bus_event_stream()

    def on_play(self):
        # Pause -> Play에서는 기존 상태 유지
        if self._subscription is not None:
            return

        self._pending = False
        self._moved = False
        self._initial_position = None

        prim = self.usd_context.get_stage().GetPrimAtPath(self.prim_path)

        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)

            for op in xform.GetOrderedXformOps():
                if (
                    op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                    and not op.IsInverseOp()
                ):
                    self._initial_position = op.Get()
                    break

        self._subscription = self._bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(EVENT_NAME),
            self._on_event,
        )

    def _on_event(self, event):
        if self._moved:
            return

        if not event.payload.get("contact", False):
            return

        # 이벤트 콜백에서는 이동 요청만 기록
        self._pending = True

    def on_update(self, current_time, delta_time):
        if not self._pending or self._moved:
            return

        self._pending = False

        prim = self.usd_context.get_stage().GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return

        xform = UsdGeom.Xformable(prim)

        for op in xform.GetOrderedXformOps():
            if (
                op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                and not op.IsInverseOp()
            ):
                current_position = op.Get()

                if current_position is None:
                    return

                # 현재 위치 + Delta
                if op.GetTypeName() == "double3":
                    new_position = (
                        Gf.Vec3d(current_position)
                        + Gf.Vec3d(*DELTA_POSITION)
                    )
                else:
                    new_position = (
                        Gf.Vec3f(current_position)
                        + Gf.Vec3f(*DELTA_POSITION)
                    )

                op.Set(new_position)

                self._moved = True

                carb.log_info(
                    f"[GuardEventMover] {self.prim_path} "
                    f"moved by {DELTA_POSITION}"
                )
                break

    def on_stop(self):
        # 이벤트 구독 해제
        self._subscription = None
        self._pending = False
        self._moved = False

        # 초기 위치 복원
        if self._initial_position is not None:
            prim = self.usd_context.get_stage().GetPrimAtPath(self.prim_path)

            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)

                for op in xform.GetOrderedXformOps():
                    if (
                        op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                        and not op.IsInverseOp()
                    ):
                        op.Set(self._initial_position)

                        carb.log_info(
                            f"[GuardEventMover] {self.prim_path} "
                            f"restored to {self._initial_position}"
                        )
                        break

        self._initial_position = None

    def on_destroy(self):
        self.on_stop()

