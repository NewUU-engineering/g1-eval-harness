"""
adapters/unitree_rl_gym_adapter.py

Reference adapter: unitree_rl_gym's pretrained G1 walking policy
(deploy/pre_train/g1/motion.pt, a 12-DOF legs-only LSTM policy).

Every constant below is copied from unitree_rl_gym's own sim2sim
(deploy/deploy_mujoco/configs/g1.yaml and deploy_mujoco.py). If your numbers
differ from the training/deploy config, the policy WILL fall - that is the #1
cause of "sim2sim doesn't work".

Observation (47) = [ang_vel*0.25 (3), projected_gravity (3), cmd*[2,2,0.25] (3),
                    (q-q0) legs (12), dq*0.05 legs (12), last_action (12), sin(phase), cos(phase)]
Action (12)      = leg joint offsets; target = q0 + 0.25*action
"""
from typing import Any, Dict, Optional
import numpy as np

from adapters.base import PolicyAdapter
from harness.types import JointCommand, RobotState
from utils.joint_mapping import CANONICAL_G1_JOINTS
from utils.policy_runtime import load_policy

# unitree_rl_gym trains the G1 in the standard Unitree / MuJoCo joint order.
UNITREE_RL_GYM_G1_ORDER = list(CANONICAL_G1_JOINTS)

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# --- g1.yaml ---
LEG_DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, dtype=np.float32)
LEG_KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float64)
LEG_KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float64)
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE = 0.25, 1.0, 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
GAIT_PERIOD_S = 0.8   # deploy_mujoco.py: period = 0.8
NUM_OBS, NUM_ACTIONS = 47, 12

# The policy was trained with the upper body welded in place (g1_12dof model).
# On the 29-DOF robot we hold waist and arms at 0 with stiff PD - this is what you
# would do on hardware too. These gains are part of THIS policy's deployment.
HOLD_GAINS = {"waist": (300.0, 5.0), "shoulder": (60.0, 2.0), "elbow": (60.0, 2.0), "wrist": (20.0, 1.0)}


class UnitreeRLGymAdapter(PolicyAdapter):
    name = "unitree_rl_gym_g1_motion"
    POLICY_JOINT_ORDER = UNITREE_RL_GYM_G1_ORDER
    policy_dt = 0.02

    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(checkpoint_path, config)
        if checkpoint_path is None:
            raise ValueError("UnitreeRLGymAdapter needs --checkpoint (e.g. checkpoints/unitree_rl_gym_g1_motion.pt)")
        self.policy = load_policy(checkpoint_path, backend=self.config.get("backend", "auto"))
        self.leg_idx = self.indices_of(LEG_JOINTS)          # positions of the legs in POLICY order

        # Full 29-DOF default pose / gains in POLICY order, then mapped to canonical once.
        q0 = np.zeros(29)
        q0[self.leg_idx] = LEG_DEFAULT
        kp, kd = np.zeros(29), np.zeros(29)
        kp[self.leg_idx], kd[self.leg_idx] = LEG_KP, LEG_KD
        for i, n in enumerate(self.POLICY_JOINT_ORDER):
            if i in self.leg_idx:
                continue
            key = next(k for k in HOLD_GAINS if k in n)
            kp[i], kd[i] = HOLD_GAINS[key]
        self.q0_policy = q0
        self.q0_canonical = self.mapper.to_source(q0)
        self.kp_canonical = self.mapper.to_source(kp)
        self.kd_canonical = self.mapper.to_source(kd)
        self.reset(0)

    # ------------------------------------------------------------------
    def default_pose(self) -> np.ndarray:
        return self.q0_canonical.copy()

    def reset(self, seed: int) -> None:
        self.policy.reset()                       # clear LSTM memory every episode
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.step = 0

    def act(self, state: RobotState) -> JointCommand:
        self.step += 1
        q = self.mapper.to_target(state.q)[self.leg_idx]       # canonical -> policy order -> legs
        dq = self.mapper.to_target(state.dq)[self.leg_idx]

        # Phase clock exactly as deploy_mujoco.py: counter counts physics steps and the policy
        # runs after the 10th, so at the k-th policy call t = k * 0.02 s.
        phase = (self.step * self.policy_dt) % GAIT_PERIOD_S / GAIT_PERIOD_S

        obs = np.concatenate([
            state.base_ang_vel * ANG_VEL_SCALE,
            state.projected_gravity,                 # [0,0,-1] upright - same as get_gravity_orientation()
            state.command * CMD_SCALE,
            (q - LEG_DEFAULT) * DOF_POS_SCALE,
            dq * DOF_VEL_SCALE,
            self.last_action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        ]).astype(np.float32)
        assert obs.shape == (NUM_OBS,), obs.shape

        action = self.policy(obs)
        self.last_action = action.copy()

        target_policy = self.q0_policy.copy()
        target_policy[self.leg_idx] += action * ACTION_SCALE
        return JointCommand(q=self.mapper.to_source(target_policy),
                            kp=self.kp_canonical, kd=self.kd_canonical)

    def describe(self) -> Dict[str, Any]:
        return {"source": "unitreerobotics/unitree_rl_gym deploy/pre_train/g1/motion.pt",
                "backend": self.policy.backend, "recurrent": self.policy.is_recurrent,
                "controlled_joints": LEG_JOINTS, "obs_dim": NUM_OBS, "gait_period_s": GAIT_PERIOD_S,
                "upper_body": "held at 0 with HOLD_GAINS"}
