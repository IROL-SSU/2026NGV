"""Attach to UR20 to sort Guard events with a selected Lula planner."""

import sys
from pathlib import Path

import carb
import carb.events
import omni.kit.app
import omni.timeline
from isaacsim.robot.surface_gripper import _surface_gripper

try:
    from omni.behavior.scripting.core import BehaviorScript
except ImportError:
    from omni.kit.scripting import BehaviorScript


# Change this value while stopped, then reload this script to switch planners.
PLANNER = "RMPflow"  # Supported: "RMPflow", "RRT".
# Set both paths for a supplied model, or leave both None to generate from USD.
URDF_PATH = None
ROBOT_DESCRIPTION_PATH = None
# Optional existing configuration files; defaults are beside this main.py.
RMPFLOW_CONFIG_PATH = None
RRT_CONFIG_PATH = None

EVENT = "ngv.product_loader.guard_contact"
ROBOT = "/World/Robot/ur20_gripper/ur20"

# Behavior Script loads an individual file, so make its sibling modules
# importable without depending on the Extension or other behavior programs.
SCRIPT_DIR = str(Path(__file__).resolve().parent.parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ur20_control.controller import UR20Controller
from ur20_control.safety import SafetyMonitor
from ur20_control.state_machine import StateMachine
from ur20_control.tasks import TaskExecutor
from ur20_control.workcell import Workcell


class UR20_BehaviorScript(BehaviorScript):
    """Receive Guard events and drive one independent sorting state machine."""

    def on_init(self):
        """Validate the attachment and subscribe to the shared message bus."""
        if str(self.prim_path) != ROBOT:
            raise ValueError("Attach UR20 main.py to " + ROBOT)
        self.machine = self.robot = self.safety = None
        self.cell = None
        self._early_events = []
        bus = omni.kit.app.get_app().get_message_bus_event_stream()
        self._subscription = bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string(EVENT), self._arrival
        )
        # Author robot physics in the session before the first PhysX Play frame.
        if omni.timeline.get_timeline_interface().is_stopped():
            self._ensure()

    def _ensure(self):
        """Initialize the selected planner and scene only when Play begins."""
        if self.machine is not None:
            return
        gripper = _surface_gripper.acquire_surface_gripper_interface()
        held = gripper.get_gripped_objects(ROBOT + "/long_gripper/SurfaceGripper")
        if held:
            raise RuntimeError("Remove the held product before reloading UR20 control")
        cell = Workcell(self.stage)
        self.cell = cell
        if cell.interrupted():
            if omni.timeline.get_timeline_interface().is_stopped():
                cell.mark_interrupted(False)
            else:
                cell.brake(True)
                raise RuntimeError(
                    "Interrupted sort: press Stop before reloading UR20 control"
                )
        config = {
            "urdf": URDF_PATH,
            "description": ROBOT_DESCRIPTION_PATH,
            "rmpflow": RMPFLOW_CONFIG_PATH,
            "rrt": RRT_CONFIG_PATH,
        }
        self.robot = UR20Controller(self.stage, cell, PLANNER, config)
        self.machine = StateMachine(cell, self.robot, TaskExecutor(cell, self.robot))
        self.safety = SafetyMonitor(self.stage, self.usd_context.get_stage_id())
        for path, kind in self._early_events:
            self.machine.arrive(path, kind)
        self._early_events.clear()

    def _arrival(self, event):
        """Validate message fields before giving an arrival to the queue."""
        if not event.payload:
            return
        path, kind = event.payload.get("prim", ""), event.payload.get("product", "")
        if self.machine is None:
            self._early_events.append((path, kind))
        else:
            self.machine.arrive(path, kind)

    def on_play(self):
        """Start or resume the state machine after required validation."""
        try:
            self._ensure()
            self.safety.play()
            self.machine.play()
        except Exception as exc:
            if self.machine is not None:
                self.machine.fail(exc)
            elif self.cell is not None:
                self.cell.brake(True)
            carb.log_error("[UR20] Could not start: " + str(exc))

    def on_update(self, current_time, delta_time):
        """Advance only the current task once per simulation update."""
        if self.machine is not None:
            if self.safety is not None:
                try:
                    self.safety.check(self.machine)
                except Exception as exc:
                    self.machine.fail(exc)
            self.machine.tick(delta_time)

    def on_stop(self):
        """Hold an attached carton safely and release idle resources."""
        if self.machine is not None:
            self.machine.stop()
        elif self.cell is not None and self.cell.interrupted():
            # A failed reload remains braked until the operator explicitly Stops.
            self.cell.mark_interrupted(False)
        self._early_events.clear()
        if self.safety is not None:
            self.safety.stop()

    def on_destroy(self):
        """Release the event subscription and planner resources."""
        self._subscription = None
        if self.machine is not None:
            self.machine.stop()
        if self.robot is not None:
            self.robot.close()
        self.machine = self.robot = self.safety = None
