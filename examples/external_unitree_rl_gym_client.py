"""
examples/external_unitree_rl_gym_client.py

Runs the unitree_rl_gym G1 policy as a SEPARATE PROCESS that talks to the
harness over adapters/external/protocol.py. Use it as the template for wrapping
a policy that lives in another environment (other venv, ROS 2 node, ...).

It also proves the external path is equivalent to the in-process path:
both produce identical metrics (see tests/test_external_parity.py).
--wire-order isaaclab sends joints in IsaacLab order to exercise the joint mapping.

  # terminal 1
  python evaluate.py --adapter adapters.external.socket_bridge:SocketBridgeAdapter --set port=7788
  # terminal 2
  python examples/external_unitree_rl_gym_client.py --checkpoint checkpoints/unitree_rl_gym_g1_motion.pt --port 7788
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from adapters.external.protocol import PolicyClient  # noqa: E402
from adapters.unitree_rl_gym_adapter import UnitreeRLGymAdapter  # noqa: E402
from harness.types import RobotState  # noqa: E402
from utils.joint_mapping import JointOrderMapper, CANONICAL_G1_JOINTS  # noqa: E402
from utils.joint_orders import ISAACLAB_G1_29DOF_ORDER  # noqa: E402


class UnitreeRLGymClient(PolicyClient):
    def __init__(self, checkpoint: str, wire_order: str, backend: str):
        self.inner = UnitreeRLGymAdapter(checkpoint, {"backend": backend})
        order = ISAACLAB_G1_29DOF_ORDER if wire_order == "isaaclab" else list(CANONICAL_G1_JOINTS)
        self.wire = JointOrderMapper(target_joint_names=order, source_joint_names=list(CANONICAL_G1_JOINTS))

    def hello(self):
        return {"name": self.inner.name, "joint_order": list(self.wire.target_names),
                "policy_dt": self.inner.policy_dt,
                "default_pose": self.wire.to_target(self.inner.default_pose()).tolist(),
                "describe": self.inner.describe()}

    def reset(self, seed):
        self.inner.reset(seed)

    def act(self, s):
        to_c = self.wire.to_source
        state = RobotState(t=s["t"], q=to_c(s["q"]), dq=to_c(s["dq"]), tau=to_c(s["tau"]),
                           base_quat=s["base_quat"], base_ang_vel=s["base_ang_vel"],
                           base_lin_acc=s["base_lin_acc"], projected_gravity=s["projected_gravity"],
                           command=s["command"], base_pos=s["base_pos"], base_lin_vel=s["base_lin_vel"])
        cmd = self.inner.act(state)
        to_w = self.wire.to_target
        return {"q": to_w(cmd.q), "kp": to_w(np.asarray(cmd.kp)), "kd": to_w(np.asarray(cmd.kd))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--wire-order", choices=["canonical", "isaaclab"], default="canonical")
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    UnitreeRLGymClient(a.checkpoint, a.wire_order, a.backend).serve(a.host, a.port)
