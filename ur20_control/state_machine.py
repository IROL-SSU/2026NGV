"""Small state dispatcher for the independent UR20 sorting behavior."""

from collections import deque

import carb


class StateMachine:
    """Own only task transitions and the FIFO of Guard arrivals."""

    def __init__(self, workcell, robot, tasks):
        """Start stopped with no queued or active products."""
        self.cell, self.robot, self.tasks = workcell, robot, tasks
        self.queue = deque()
        self.seen = set()
        self.state = "Stopped"
        self.elapsed = self.stable = 0.0
        self.resume_check = False
        self._initialize_home = False
        self.handlers = {
            "Homing": self._homing,
            "Waiting": self._waiting,
            "Picking": self._picking,
            "Carrying": self._carrying,
            "Returning": self._returning,
        }

    def _transition(self, state):
        """Enter a state and reset its timeout counters."""
        self.state = state
        self.elapsed = self.stable = 0.0
        carb.log_info("[UR20] " + state)

    def arrive(self, path, kind):
        """Validate, deduplicate, and queue an arrival in FIFO order."""
        product = self.cell.product(path)
        if not product or product.GetCustomDataByKey("ngv:product") != kind:
            return
        path = str(product.GetPath())
        if path in self.seen or product.GetCustomDataByKey("ngv:sortedLane"):
            return
        self.seen.add(path)
        self.queue.append(path)
        self.cell.brake(True)

    def play(self):
        """Start from Home or revalidate a paused transaction."""
        if self.state == "Stopped":
            if self.tasks.target and self.robot.holding(self.tasks.target):
                raise RuntimeError("Remove the held product manually before restart")
            self.cell.mark_interrupted(False)
            self.tasks.finish()
            self._initialize_home = True
            self._transition("Homing")
        elif self.state != "Error":
            self.resume_check = True

    def stop(self):
        """Hold a carried product and brake Lane01; otherwise reset safely."""
        interrupted = bool(
            self.tasks.target
            and self.state in ("Picking", "Carrying", "Returning")
        )
        holding = bool(self.tasks.target and self.robot.holding(self.tasks.target))
        self.robot.hold()
        self.queue.clear()
        self.seen.clear()
        self._transition("Stopped")
        if interrupted:
            self.cell.mark_interrupted(True)
        if holding or interrupted or self.cell.interrupted():
            self.cell.brake(True)
            if holding:
                carb.log_error(
                    "[UR20] Stop while holding product: remove it manually before restart"
                )
        else:
            self.tasks.finish()
            self.cell.brake(False)

    def tick(self, dt):
        """Call only the action for the current state and stop on failure."""
        if self.state in ("Stopped", "Error") or dt <= 0:
            return
        dt = min(float(dt), 0.05)
        try:
            if self.resume_check:
                if self.tasks.target and not self.cell.product(self.tasks.target):
                    raise RuntimeError("Active product vanished during Pause")
                if self.tasks.target and self.tasks.phase not in (
                    "settle",
                    "approach",
                    "descend",
                    "attach",
                ):
                    self.tasks._require_hold()
                self.robot.goal_key = None
                self.resume_check = False
            self.elapsed += dt
            self.handlers[self.state](dt)
        except Exception as exc:
            self.fail(exc)

    def fail(self, error):
        """Hold the arm, retain suction, and brake the infeed on errors."""
        try:
            self.robot.hold()
        finally:
            self.cell.brake(True)
            self._transition("Error")
            carb.log_error("[UR20] ERROR: " + str(error))

    def _homing(self, dt):
        """Reach initial USD Home before accepting queued work."""
        if self._initialize_home:
            self.robot.initialize_home()
            self._initialize_home = False
            return
        self.stable = self.stable + dt if self.robot.move_home(dt) else 0.0
        if self.stable >= 0.25:
            self.robot.suction(False)
            self._transition("Waiting")
        elif self.elapsed > 45:
            raise RuntimeError("Initial Home timed out")

    def _waiting(self, dt):
        """Take the next still-active product, or release the input belt."""
        if not self.queue:
            self.cell.brake(False)
            return
        self.cell.brake(True)
        while self.queue:
            if self.tasks.begin(self.queue.popleft()):
                self._transition("Picking")
                return
        self.cell.brake(False)

    def _picking(self, dt):
        """Delegate Guard pickup and lift to the task executor."""
        if self.tasks.pick(dt):
            self._transition("Carrying")

    def _carrying(self, dt):
        """Delegate lane transfer, release, and clearance to the task executor."""
        if self.tasks.carry(dt):
            self._transition("Returning")

    def _returning(self, dt):
        """Return Home before allowing another queued product."""
        self.stable = self.stable + dt if self.robot.move_home(dt) else 0.0
        if self.stable >= 0.25:
            self.tasks.finish()
            self.cell.brake(bool(self.queue))
            self._transition("Waiting")
        elif self.elapsed > 45:
            raise RuntimeError("Return Home timed out")
