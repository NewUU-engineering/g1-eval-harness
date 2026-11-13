"""Marvarid's mapper: round trips and a known permutation."""
import numpy as np
import pytest
from utils.joint_mapping import JointOrderMapper, CANONICAL_G1_JOINTS, UNITREE_DDS_ORDER
from utils.joint_orders import ISAACLAB_G1_29DOF_ORDER


def test_orders_are_permutations_of_canonical():
    for order in (UNITREE_DDS_ORDER, ISAACLAB_G1_29DOF_ORDER):
        assert sorted(order) == sorted(CANONICAL_G1_JOINTS) and len(order) == 29


def test_round_trip_isaaclab():
    m = JointOrderMapper(target_joint_names=ISAACLAB_G1_29DOF_ORDER)
    x = np.arange(29.0)
    assert np.array_equal(m.to_source(m.to_target(x)), x)
    y = m.to_target(x)
    for i, name in enumerate(ISAACLAB_G1_29DOF_ORDER):        # value follows its joint name
        assert y[i] == CANONICAL_G1_JOINTS.index(name)


def test_batch_dims():
    m = JointOrderMapper(target_joint_names=ISAACLAB_G1_29DOF_ORDER)
    X = np.random.default_rng(0).normal(size=(5, 29))
    assert np.allclose(m.to_source(m.to_target(X)), X)


def test_mismatched_names_raise():
    with pytest.raises(ValueError):
        JointOrderMapper(target_joint_names=ISAACLAB_G1_29DOF_ORDER[:-1] + ["bogus_joint"])
