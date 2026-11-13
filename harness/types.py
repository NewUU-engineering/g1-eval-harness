"""
harness/types.py

The two messages exchanged between the harness and every policy.
They deliberately mirror Unitree's LowState / LowCmd, so an in-process
adapter and an external DDS stack see the same thing.

All joint arrays are in CANONICAL order (utils.joint_mapping.CANONICAL_G1_JOINTS,
which is also the G1 MuJoCo order and the Unitree motor/DDS order).
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

NUM_JOINTS = 29


@dataclass
class RobotState:
    t: float                    # seconds since episode start (sim time)
    q: np.ndarray               # [29] joint positions (rad)
    dq: np.ndarray              # [29] joint velocities (rad/s)
    tau: np.ndarray             # [29] torque applied in the last physics step (N*m)
    base_quat: np.ndarray       # [4] pelvis orientation, world frame, (w, x, y, z)
    base_ang_vel: np.ndarray    # [3] pelvis angular velocity, BODY frame (what the IMU gyro measures)
    base_lin_acc: np.ndarray    # [3] pelvis accelerometer, BODY frame (includes gravity, like a real IMU)
    projected_gravity: np.ndarray  # [3] unit gravity vector in BODY frame ([0,0,-1] when upright)
    command: np.ndarray         # [3] velocity command (vx, vy, yaw_rate) in the robot's heading frame
    # --- Privileged (not available on the real robot). Use only if your policy was trained with them,
    #     and say so in describe(); it is recorded in the results.
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))      # world
    base_lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # BODY frame

    def copy(self) -> "RobotState":
        return RobotState(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})


@dataclass
class JointCommand:
    """Per-joint PD target, exactly like one LowCmd. Harness computes
    tau = kp*(q - q_meas) + kd*(dq - dq_meas) + tau_ff, then clips to the motor limit."""
    q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    dq: Optional[np.ndarray] = None
    tau_ff: Optional[np.ndarray] = None

    def validated(self) -> "JointCommand":
        def arr(x, name, default=0.0):
            a = np.full(NUM_JOINTS, default, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64).reshape(-1)
            if a.shape != (NUM_JOINTS,):
                raise ValueError(f"JointCommand.{name} must have shape ({NUM_JOINTS},), got {a.shape}")
            return a
        out = JointCommand(q=arr(self.q, "q"), kp=arr(self.kp, "kp"), kd=arr(self.kd, "kd"),
                           dq=arr(self.dq, "dq"), tau_ff=arr(self.tau_ff, "tau_ff"))
        if np.any(out.kp < 0) or np.any(out.kd < 0):
            raise ValueError("JointCommand gains must be non-negative.")
        return out

    def is_finite(self) -> bool:
        return all(np.all(np.isfinite(a)) for a in (self.q, self.kp, self.kd, self.dq, self.tau_ff))
