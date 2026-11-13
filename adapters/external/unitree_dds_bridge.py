"""
adapters/external/unitree_dds_bridge.py

Evaluate a policy that runs inside its OWN deployment stack and talks to the
robot over Unitree DDS (rt/lowstate -> policy -> rt/lowcmd). Examples: SONIC's
C++ deploy, BeyondMimic's ROS 2 controller when configured for the Unitree SDK
interface, or anything written against unitree_sdk2.

The harness pretends to be the robot (same idea as unitreerobotics/unitree_mujoco):
  * every physics step it publishes LowState built from the MuJoCo state,
  * it applies the most recent LowCmd with the SAME PD law, torque limits,
    scenarios and metrics as every other policy.

Because the external stack runs on its own clock, the simulation is paced to
wall-clock time (realtime_factor=1.0). Results are therefore NOT bit-reproducible
like in-process or socket runs; run more seeds and report latency stats
(saved under adapter.describe in summary.json).

STATUS: written against the unitree_sdk2_python API but NOT yet tested against a
real SONIC / BeyondMimic deployment. Expect to adjust: network interface, domain
id, mode_machine, and whatever start-up handshake the stack expects (many wait for
a remote-controller button in wireless_remote before sending commands).

  pip install unitree_sdk2py   (needs cyclonedds; see unitree_sdk2_python README)

  python evaluate.py --adapter adapters.external.unitree_dds_bridge:UnitreeDDSBridgeAdapter \
      --set interface=lo --set domain_id=1 --scenarios flat_nominal --seeds 0
  # then start the external stack pointed at the same interface/domain
"""
import math
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from adapters.base import PolicyAdapter
from harness.config import G1_REFERENCE_GAINS
from harness.types import JointCommand, RobotState
from utils.joint_mapping import CANONICAL_G1_JOINTS, UNITREE_DDS_ORDER

NUM_MOTORS_HG = 35   # unitree_hg LowState/LowCmd carry 35 motor slots; the G1 29-DOF uses 0..28


def _rpy_from_quat(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class UnitreeDDSBridgeAdapter(PolicyAdapter):
    name = "external_unitree_dds"
    POLICY_JOINT_ORDER = UNITREE_DDS_ORDER     # motor index order on the wire
    command_rate = "physics"                   # act() every physics step: publish state, read latest cmd

    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(checkpoint_path, config)
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as e:
            raise ImportError("UnitreeDDSBridgeAdapter needs unitree_sdk2py (pip install unitree_sdk2py).") from e

        c = self.config
        self.realtime_factor = float(c.get("realtime_factor", 1.0))
        self.cmd_timeout_s = float(c.get("cmd_timeout_s", 0.1))
        self.mode_machine = int(c.get("mode_machine", 0))
        self.lowstate_topic = c.get("lowstate_topic", "rt/lowstate")
        self.lowcmd_topic = c.get("lowcmd_topic", "rt/lowcmd")
        self.reset_command = c.get("reset_command")          # optional shell cmd run at every reset
        self.startup_wait_s = float(c.get("startup_wait_s", 0.0))
        dp = c.get("default_pose")
        self._default_pose = (self.mapper.to_source(np.asarray(dp, dtype=float)) if dp is not None
                              else np.zeros(29))
        self.hold_kp = np.array([G1_REFERENCE_GAINS[n][0] for n in CANONICAL_G1_JOINTS])
        self.hold_kd = np.array([G1_REFERENCE_GAINS[n][1] for n in CANONICAL_G1_JOINTS])

        ChannelFactoryInitialize(int(c.get("domain_id", 1)), c.get("interface", "lo"))
        self._crc = CRC()
        self._msg = unitree_hg_msg_dds__LowState_()
        self._pub = ChannelPublisher(self.lowstate_topic, LowState_)
        self._pub.Init()
        self._lock = threading.Lock()
        self._latest = None
        self._latest_wall = 0.0
        self._sub = ChannelSubscriber(self.lowcmd_topic, LowCmd_)
        self._sub.Init(self._on_lowcmd, 10)

        self.stats = {"cmds_received": 0, "stale_steps": 0, "steps_before_first_cmd": 0, "max_cmd_age_ms": 0.0}

    # --------------------------------------------------------------- DDS
    def _on_lowcmd(self, msg):
        with self._lock:
            self._latest = msg
            self._latest_wall = time.perf_counter()
            self.stats["cmds_received"] += 1

    def _publish(self, s: RobotState):
        m = self._msg
        q, dq, tau = (self.mapper.to_target(a) for a in (s.q, s.dq, s.tau))
        for i in range(29):
            ms = m.motor_state[i]
            ms.q, ms.dq, ms.tau_est = float(q[i]), float(dq[i]), float(tau[i])
        w, x, y, z = (float(v) for v in s.base_quat)
        m.imu_state.quaternion = [w, x, y, z]
        m.imu_state.gyroscope = [float(v) for v in s.base_ang_vel]
        m.imu_state.accelerometer = [float(v) for v in s.base_lin_acc]
        m.imu_state.rpy = list(_rpy_from_quat(w, x, y, z))
        m.mode_machine = self.mode_machine
        m.tick = int(round(s.t * 1000))
        m.crc = self._crc.Crc(m)
        self._pub.Write(m)

    # --------------------------------------------------------------- adapter API
    def default_pose(self) -> np.ndarray:
        return self._default_pose.copy()

    def reset(self, seed: int) -> None:
        if self.reset_command:
            subprocess.run(shlex.split(self.reset_command), check=False)
        with self._lock:
            self._latest = None
        if self.startup_wait_s > 0:
            time.sleep(self.startup_wait_s)
        self._wall0 = None

    def act(self, state: RobotState) -> JointCommand:
        now = time.perf_counter()
        if self._wall0 is None:
            self._wall0 = now - state.t / self.realtime_factor
        lag = (self._wall0 + state.t / self.realtime_factor) - now
        if lag > 0:
            time.sleep(lag)                    # pace the simulation to wall clock

        self._publish(state)

        with self._lock:
            msg, t_rx = self._latest, self._latest_wall
        if msg is None:
            self.stats["steps_before_first_cmd"] += 1
            return JointCommand(q=self._default_pose, kp=self.hold_kp, kd=self.hold_kd)
        age = time.perf_counter() - t_rx
        self.stats["max_cmd_age_ms"] = max(self.stats["max_cmd_age_ms"], age * 1e3)
        if age > self.cmd_timeout_s:
            self.stats["stale_steps"] += 1
        mc = msg.motor_cmd
        arr = lambda f: self.mapper.to_source(np.array([getattr(mc[i], f) for i in range(29)], dtype=float))
        return JointCommand(q=arr("q"), dq=arr("dq"), kp=arr("kp"), kd=arr("kd"), tau_ff=arr("tau"))

    def describe(self) -> Dict[str, Any]:
        return {"transport": "unitree-dds-realtime", "realtime_factor": self.realtime_factor,
                "topics": [self.lowstate_topic, self.lowcmd_topic], "stats": dict(self.stats)}

    def close(self) -> None:
        try:
            self._sub.Close()
        except Exception:
            pass
