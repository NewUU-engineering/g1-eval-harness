"""
adapters/external/protocol.py

Wire protocol between the harness and a policy running in ANOTHER PROCESS
(another venv, a ROS 2 node, a C++ program, another machine).

This file is standalone (only numpy + stdlib) on purpose: copy it into your
policy's environment and use PolicyClient, or re-implement it in C++ - it is
just length-prefixed frames over TCP.

Frame:   4-byte ASCII type | uint32 little-endian payload length | payload
Types:
  HELO  client -> harness   JSON  {"name", "joint_order":[29 names], "policy_dt", "default_pose":[29], "describe":{}}
  RSET  harness -> client   JSON  {"seed": int}
  RACK  client -> harness   JSON  {}
  STAT  harness -> client   float64[STATE_LEN]   (joints in the CLIENT's joint order)
  CMND  client -> harness   float64[CMD_LEN]     (joints in the CLIENT's joint order)
  BYE_  harness -> client   JSON  {}

STATE layout (float64):  t | q[29] | dq[29] | tau[29] | base_quat_wxyz[4] | base_ang_vel_body[3]
                         | base_lin_acc_body[3] | projected_gravity[3] | command[3]
                         | base_pos_world[3] | base_lin_vel_body[3]          (last two are privileged)
CMD layout (float64):    q[29] | dq[29] | kp[29] | kd[29] | tau_ff[29]

The harness is lock-step: simulation time does not advance until CMND arrives,
so results are deterministic and identical to running the policy in-process.
"""
import json
import socket
import struct
from typing import Dict, Tuple

import numpy as np

N = 29
STATE_FIELDS = [("t", 1), ("q", N), ("dq", N), ("tau", N), ("base_quat", 4), ("base_ang_vel", 3),
                ("base_lin_acc", 3), ("projected_gravity", 3), ("command", 3),
                ("base_pos", 3), ("base_lin_vel", 3)]
CMD_FIELDS = [("q", N), ("dq", N), ("kp", N), ("kd", N), ("tau_ff", N)]
STATE_LEN = sum(n for _, n in STATE_FIELDS)
CMD_LEN = sum(n for _, n in CMD_FIELDS)
JOINT_FIELDS_STATE = {"q", "dq", "tau"}


def pack(fields, values: Dict[str, np.ndarray]) -> bytes:
    return np.concatenate([np.asarray(values[k], dtype=np.float64).reshape(-1) for k, _ in fields]).tobytes()


def unpack(fields, payload: bytes) -> Dict[str, np.ndarray]:
    a = np.frombuffer(payload, dtype=np.float64)
    out, i = {}, 0
    for k, n in fields:
        out[k] = a[i:i + n].copy()
        i += n
    if i != a.size:
        raise ValueError(f"Bad payload length {a.size}, expected {i}")
    out["t"] = float(out["t"][0]) if "t" in out else None
    return out


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def send_frame(sock: socket.socket, kind: str, payload: bytes) -> None:
    assert len(kind) == 4
    sock.sendall(kind.encode("ascii") + struct.pack("<I", len(payload)) + payload)


def recv_frame(sock: socket.socket) -> Tuple[str, bytes]:
    head = _recv_exact(sock, 8)
    kind, length = head[:4].decode("ascii"), struct.unpack("<I", head[4:])[0]
    return kind, _recv_exact(sock, length)


def send_json(sock, kind, obj):
    send_frame(sock, kind, json.dumps(obj).encode("utf-8"))


def recv_json(sock, expect: str):
    kind, payload = recv_frame(sock)
    if kind != expect:
        raise ConnectionError(f"expected {expect}, got {kind}")
    return json.loads(payload.decode("utf-8"))


class PolicyClient:
    """Client side. Subclass and implement hello(), reset(seed), act(state) -> dict(q, kp, kd[, dq, tau_ff]).
    All joint arrays are in YOUR joint order (the one you send in hello())."""

    def hello(self) -> Dict:
        raise NotImplementedError

    def reset(self, seed: int) -> None:
        raise NotImplementedError

    def act(self, state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def serve(self, host: str = "127.0.0.1", port: int = 7788, connect_timeout_s: float = 30.0) -> None:
        import time
        deadline = time.time() + connect_timeout_s
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=5.0)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_json(sock, "HELO", self.hello())
        try:
            while True:
                kind, payload = recv_frame(sock)
                if kind == "RSET":
                    self.reset(int(json.loads(payload)["seed"]))
                    send_json(sock, "RACK", {})
                elif kind == "STAT":
                    cmd = self.act(unpack(STATE_FIELDS, payload))
                    zeros = np.zeros(N)
                    cmd = {"q": cmd["q"], "kp": cmd["kp"], "kd": cmd["kd"],
                           "dq": cmd.get("dq", zeros), "tau_ff": cmd.get("tau_ff", zeros)}
                    send_frame(sock, "CMND", pack(CMD_FIELDS, cmd))
                elif kind == "BYE_":
                    break
                else:
                    raise ConnectionError(f"unexpected frame {kind}")
        finally:
            sock.close()
