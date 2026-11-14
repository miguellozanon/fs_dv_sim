"""Monotonic ROS timeline over a simulator clock that can restart.

fsdsim's time starts at 0 on every Isaac Sim run and on every /sim/reset. If ROS
stamps followed it, every TF listener would break:
  * on wall time (RViz default), tf2 rejects the new stamps as TF_OLD_DATA, since they
    are older than what it already holds -> "Frame [map] does not exist" / nothing drawn;
  * on sim time, /clock jumps back and every listener wipes its buffer.

So the ROS side publishes `ros_t = sim_t + offset`. When the simulator restarts, the
offset is raised so that ROS time continues 1 s after the last published stamp.
"""

from __future__ import annotations


class MonotonicTimeline:
    GAP = 1.0        # ROS time skipped across a restart (marks the boundary clearly)
    REORDER = 0.5    # smaller backward steps are out-of-order frames, not a restart

    def __init__(self, log=None):
        self.offset = 0.0
        self.last = None          # last ROS time handed out by advance()
        self._new_session = False
        self._log = log or (lambda _m: None)

    def new_session(self) -> None:
        """A new simulator connected: its clock restarts, even after a very short previous run."""
        self._new_session = True

    def advance(self, sim_t: float) -> float:
        """Call once per simulator frame (before converting its stamps). Returns ROS time."""
        ros_t = sim_t + self.offset
        if self.last is not None and (ros_t < self.last - self.REORDER or (self._new_session and ros_t <= self.last)):
            self.offset = self.last + self.GAP - sim_t
            self._log(f"simulator time restarted (t={sim_t:.2f} s); ROS time continues at {self.last + self.GAP:.2f} s "
                      "so TF buffers in RViz and other nodes stay valid")
            ros_t = sim_t + self.offset
        self._new_session = False
        if self.last is None or ros_t > self.last:
            self.last = ros_t
        return ros_t

    def to_ros(self, sim_t: float) -> float:
        return sim_t + self.offset

    def to_sim(self, ros_t: float) -> float:
        return ros_t - self.offset
