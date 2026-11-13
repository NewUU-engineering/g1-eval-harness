"""
adapters/external/socket_bridge.py

Evaluate a policy that runs in another process, lock-step over TCP.
The harness side of adapters/external/protocol.py.

  python evaluate.py --adapter adapters.external.socket_bridge:SocketBridgeAdapter \
      --set port=7788 \
      --set launch="python examples/external_unitree_rl_gym_client.py --checkpoint checkpoints/unitree_rl_gym_g1_motion.pt --port 7788"

`launch` is optional: if given, the harness starts the client process for you;
otherwise start your client yourself (it retries the connection for 30 s).

Joint order: the client announces its own order in HELO; Marvarid's mapper
converts both ways, so the client never sees canonical order unless it wants to.
"""
import json
import shlex
import socket
import subprocess
from typing import Any, Dict, Optional

import numpy as np

from adapters.base import PolicyAdapter
from adapters.external import protocol as P
from harness.types import JointCommand, RobotState
from utils.joint_mapping import JointOrderMapper, CANONICAL_G1_JOINTS


class SocketBridgeAdapter(PolicyAdapter):
    name = "external_socket"
    command_rate = "control"

    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(checkpoint_path, config)
        host = self.config.get("host", "127.0.0.1")
        port = int(self.config.get("port", 7788))
        self.timeout_s = float(self.config.get("timeout_s", 10.0))
        self.proc = None

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.settimeout(float(self.config.get("connect_timeout_s", 60.0)))

        launch = self.config.get("launch")
        if launch:
            self.proc = subprocess.Popen(shlex.split(launch) if isinstance(launch, str) else launch)
        print(f"[SocketBridge] waiting for external policy on {host}:{port} ...")
        self.sock, _ = self.server.accept()
        self.sock.settimeout(self.timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        hello = P.recv_json(self.sock, "HELO")
        self.remote = hello
        self.name = f"external:{hello.get('name', 'unnamed')}"
        self.policy_dt = float(hello.get("policy_dt", 0.02))
        self.uses_privileged_state = bool(hello.get("uses_privileged_state", False))
        order = hello["joint_order"]
        self.POLICY_JOINT_ORDER = list(order)
        self.mapper = JointOrderMapper(target_joint_names=order, source_joint_names=list(CANONICAL_G1_JOINTS))
        self._default_pose = self.mapper.to_source(np.asarray(hello["default_pose"], dtype=np.float64))
        print(f"[SocketBridge] connected: {self.name} (policy_dt={self.policy_dt})")

    def default_pose(self) -> np.ndarray:
        return self._default_pose.copy()

    def reset(self, seed: int) -> None:
        P.send_json(self.sock, "RSET", {"seed": int(seed)})
        P.recv_json(self.sock, "RACK")

    def act(self, state: RobotState) -> JointCommand:
        values = {k: (self.mapper.to_target(v) if k in P.JOINT_FIELDS_STATE else v)
                  for k, v in state.__dict__.items()}
        values["t"] = np.array([state.t])
        P.send_frame(self.sock, "STAT", P.pack(P.STATE_FIELDS, values))
        kind, payload = P.recv_frame(self.sock)
        if kind != "CMND":
            raise ConnectionError(f"expected CMND, got {kind}")
        c = P.unpack(P.CMD_FIELDS, payload)
        to_c = self.mapper.to_source
        return JointCommand(q=to_c(c["q"]), dq=to_c(c["dq"]), kp=to_c(c["kp"]), kd=to_c(c["kd"]),
                            tau_ff=to_c(c["tau_ff"]))

    def describe(self) -> Dict[str, Any]:
        return {"transport": "tcp-lockstep", "remote": {k: v for k, v in self.remote.items() if k != "default_pose"}}

    def close(self) -> None:
        try:
            P.send_json(self.sock, "BYE_", {})
        except Exception:
            pass
        for s in (getattr(self, "sock", None), getattr(self, "server", None)):
            try:
                s.close()
            except Exception:
                pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
