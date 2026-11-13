"""Engine invariants: model sanity, determinism, seeds, physics checks."""
import numpy as np
import pytest
from dataclasses import replace

from adapters.base import PolicyAdapter
from harness.config import HarnessConfig, G1_REFERENCE_GAINS
from harness.engine import SimulationEngine
from harness.scenarios import TEST_SCENARIOS
from harness.types import JointCommand
from utils.joint_mapping import CANONICAL_G1_JOINTS

KP = np.array([G1_REFERENCE_GAINS[n][0] for n in CANONICAL_G1_JOINTS])
KD = np.array([G1_REFERENCE_GAINS[n][1] for n in CANONICAL_G1_JOINTS])


# Stiff legs: with no balance controller the G1 only stays up if hips/knees/ankles are stiff
# (Unitree's soft 40 N*m/rad ankle gains let it topple forward - expected, not a harness bug).
STIFF_KP, STIFF_KD = KP.copy(), KD.copy()
for _i, _n in enumerate(CANONICAL_G1_JOINTS):
    if any(k in _n for k in ("hip", "knee", "ankle")):
        STIFF_KP[_i], STIFF_KD[_i] = 300.0, 5.0


class HoldPose(PolicyAdapter):
    """No policy: hold the zero pose with stiff PD."""
    name = "hold_pose"

    def default_pose(self):
        return np.zeros(29)

    def reset(self, seed):
        pass

    def act(self, state):
        return JointCommand(q=self.default_pose(), kp=STIFF_KP, kd=STIFF_KD)


@pytest.fixture(scope="module")
def engine():
    return SimulationEngine(HarnessConfig())


def short(name, T=2.0):
    return replace(TEST_SCENARIOS[name], duration_s=T)


def test_model_is_real_g1(engine):
    r = engine.robot
    assert 30 < r.describe()["total_mass_kg"] < 40
    assert list(r.joint_names) == list(CANONICAL_G1_JOINTS)
    knee = CANONICAL_G1_JOINTS.index("left_knee_joint")
    assert r.q_lower[knee] < 0.0 < r.q_upper[knee] and r.q_upper[knee] > 2.0


def test_robot_can_stand_without_policy(engine):
    res = engine.run_trial(HoldPose(), short("stand_still", 3.0), 0, np.zeros(3))
    assert not res.metrics.fall, res.metrics.fall_reason


def test_same_seed_is_bit_identical(engine):
    a = engine.run_trial(HoldPose(), short("push_lateral_medium", 5.0), 7, np.zeros(3)).metrics.to_dict()
    b = engine.run_trial(HoldPose(), short("push_lateral_medium", 5.0), 7, np.zeros(3)).metrics.to_dict()
    a.pop("mean_policy_latency_ms"); b.pop("mean_policy_latency_ms")
    assert a == b


def test_different_seeds_differ(engine):
    a = engine.run_trial(HoldPose(), short("flat_nominal"), 0, np.zeros(3))
    b = engine.run_trial(HoldPose(), short("flat_nominal"), 1, np.zeros(3))
    assert not np.allclose(a.log["q"], b.log["q"])


def test_friction_is_restored(engine):
    before = engine.model.geom_friction.copy()
    engine.run_trial(HoldPose(), short("ice", 0.2), 0, np.zeros(3))
    assert np.array_equal(before, engine.model.geom_friction)


def test_torque_saturation_is_counted(engine):
    class Saturate(HoldPose):
        def act(self, state):
            return JointCommand(q=self.default_pose(), kp=STIFF_KP, kd=STIFF_KD, tau_ff=np.full(29, 1e4))
    m = engine.run_trial(Saturate(), short("flat_nominal", 0.1), 0, np.zeros(3)).metrics
    assert m.torque_saturation_rate == 1.0


def test_nan_action_is_a_failure(engine):
    class Broken(HoldPose):
        def act(self, state):
            return JointCommand(q=np.full(29, np.nan), kp=KP, kd=KD)
    m = engine.run_trial(Broken(), short("flat_nominal", 0.5), 0, np.zeros(3)).metrics
    assert m.fall and m.fall_reason == "invalid_action"


def test_policy_dt_mismatch_is_rejected(engine):
    class Slow(HoldPose):
        policy_dt = 0.01
    with pytest.raises(ValueError):
        engine.run_trial(Slow(), short("flat_nominal", 0.1), 0, np.zeros(3))
