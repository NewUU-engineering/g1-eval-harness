"""The shipped example adapter must walk; the external path must match it exactly."""
import os
import socket
import subprocess
import sys
import numpy as np
import pytest
from dataclasses import replace

from harness.config import HarnessConfig
from harness.engine import SimulationEngine
from harness.scenarios import TEST_SCENARIOS
from adapters.unitree_rl_gym_adapter import UnitreeRLGymAdapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "unitree_rl_gym_g1_motion.pt")


@pytest.fixture(scope="module")
def engine():
    return SimulationEngine(HarnessConfig())


def test_unitree_policy_walks(engine):
    res = engine.run_trial(UnitreeRLGymAdapter(CKPT), TEST_SCENARIOS["flat_nominal"], 0, np.array([0.5, 0, 0]))
    m = res.metrics
    assert not m.fall, m.fall_reason
    assert 0.35 < m.mean_forward_vel < 0.75
    assert m.lin_vel_xy_rmse < 0.2


def test_lstm_memory_reset_between_episodes(engine):
    a = UnitreeRLGymAdapter(CKPT)
    scen = replace(TEST_SCENARIOS["flat_nominal"], duration_s=2.0)
    first = engine.run_trial(a, scen, 0, np.array([0.5, 0, 0])).log["q_target"]
    second = engine.run_trial(a, scen, 0, np.array([0.5, 0, 0])).log["q_target"]
    assert np.array_equal(first, second)


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_external_socket_matches_in_process(engine):
    from adapters.external.socket_bridge import SocketBridgeAdapter
    scen = replace(TEST_SCENARIOS["push_lateral_medium"], duration_s=5.0)
    ref = engine.run_trial(UnitreeRLGymAdapter(CKPT), scen, 3, np.array([0.5, 0, 0]))
    port = _free_port()
    launch = [sys.executable, os.path.join(ROOT, "examples", "external_unitree_rl_gym_client.py"),
              "--checkpoint", CKPT, "--port", str(port), "--wire-order", "isaaclab"]
    ext = SocketBridgeAdapter(config={"port": port, "launch": launch})
    try:
        out = engine.run_trial(ext, scen, 3, np.array([0.5, 0, 0]))
    finally:
        ext.close()
    assert np.array_equal(ref.log["q"], out.log["q"])
    assert np.array_equal(ref.log["q_target"], out.log["q_target"])
