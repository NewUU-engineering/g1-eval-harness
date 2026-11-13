"""
adapters/template_adapter.py  ->  copy to adapters/<your_name>_adapter.py

Fill in every TODO. Compare against adapters/unitree_rl_gym_adapter.py,
which is a complete, working example.

Checklist (each of these has broken a sim2sim before):
  [ ] POLICY_JOINT_ORDER is exactly the order your network uses (check the training config / ONNX metadata)
  [ ] default joint angles, action scale, obs scales are copied from the TRAINING config, not guessed
  [ ] projected gravity sign: harness gives [0, 0, -1] when upright (legged_gym / IsaacLab convention)
  [ ] command scaling matches training
  [ ] phase clock period / reference-motion timing matches training
  [ ] recurrent memory is cleared in reset(); never call the policy with dummy inputs
  [ ] kp / kd are the gains the policy was trained (and is deployed) with
  [ ] policy_dt matches the harness control rate (0.02 s)
"""
from typing import Any, Dict, Optional
import numpy as np

from adapters.base import PolicyAdapter
from harness.types import JointCommand, RobotState
from utils.joint_mapping import CANONICAL_G1_JOINTS
from utils.policy_runtime import load_policy


class TemplateAdapter(PolicyAdapter):
    name = "my_policy"                                   # TODO
    POLICY_JOINT_ORDER = list(CANONICAL_G1_JOINTS)       # TODO: your network's joint order (29 names)
    policy_dt = 0.02

    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(checkpoint_path, config)
        self.policy = load_policy(checkpoint_path)       # TODO: or onnxruntime.InferenceSession(...)
        self.q0 = np.zeros(29)                           # TODO: default angles, POLICY order
        self.kp = np.full(29, 50.0)                      # TODO: training gains, POLICY order
        self.kd = np.full(29, 2.0)
        self.action_scale = 0.25                         # TODO
        self.reset(0)

    def default_pose(self) -> np.ndarray:
        return self.mapper.to_source(self.q0)            # harness wants CANONICAL order

    def reset(self, seed: int) -> None:
        self.policy.reset()
        self.last_action = np.zeros(29, dtype=np.float32)

    def act(self, state: RobotState) -> JointCommand:
        q = self.mapper.to_target(state.q)               # canonical -> your order
        dq = self.mapper.to_target(state.dq)
        obs = np.concatenate([                           # TODO: your exact observation layout
            state.base_ang_vel, state.projected_gravity, state.command,
            q - self.q0, dq, self.last_action,
        ]).astype(np.float32)
        action = self.policy(obs)
        self.last_action = action
        target = self.q0 + self.action_scale * action    # your order
        return JointCommand(q=self.mapper.to_source(target),          # back to canonical
                            kp=self.mapper.to_source(self.kp),
                            kd=self.mapper.to_source(self.kd))

    # Motion-tracking policies: uncomment so the harness can score tracking error.
    # def reference(self, t):
    #     return {"q": self.mapper.to_source(self.motion_q_at(t))}
