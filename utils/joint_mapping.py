"""
utils/joint_mapping.py
Joint Order Mapper - Coordinated with Marvarid.

Translates joint arrays between policy order and the standard MuJoCo order.
"""
from typing import List
import numpy as np

# 1. The Canonical (Standard) 29-DOF G1 Joint Order in MuJoCo
# DO NOT CHANGE THIS ORDER: This matches the MuJoCo XML kinematic tree.
CANONICAL_G1_JOINTS: List[str] = [
    # Left Leg (6 joints)
    "left_hip_pitch_joint", 
    "left_hip_roll_joint", 
    "left_hip_yaw_joint",
    "left_knee_joint", 
    "left_ankle_pitch_joint", 
    "left_ankle_roll_joint",
    
    # Right Leg (6 joints)
    "right_hip_pitch_joint", 
    "right_hip_roll_joint", 
    "right_hip_yaw_joint",
    "right_knee_joint", 
    "right_ankle_pitch_joint", 
    "right_ankle_roll_joint",
    
    # Waist / Torso (3 joints)
    "waist_yaw_joint", 
    "waist_roll_joint", 
    "waist_pitch_joint",
    
    # Left Arm (7 joints)
    "left_shoulder_pitch_joint", 
    "left_shoulder_roll_joint", 
    "left_shoulder_yaw_joint",
    "left_elbow_joint", 
    "left_wrist_roll_joint", 
    "left_wrist_pitch_joint", 
    "left_wrist_yaw_joint",
    
    # Right Arm (7 joints)
    "right_shoulder_pitch_joint", 
    "right_shoulder_roll_joint", 
    "right_shoulder_yaw_joint",
    "right_elbow_joint", 
    "right_wrist_roll_joint", 
    "right_wrist_pitch_joint", 
    "right_wrist_yaw_joint"
]

# 2. Hardware Unitree DDS Order (used by policies like SONIC or low-level SDK)
UNITREE_DDS_ORDER: List[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]


class JointOrderMapper:
    """
    Rearranges arrays between different joint orders.
    """
    def __init__(self, target_joint_names: List[str], source_joint_names: List[str] = CANONICAL_G1_JOINTS):
        self.source_names = list(source_joint_names)
        self.target_names = list(target_joint_names)

        # Safety check: make sure both lists have the exact same joints
        if set(self.source_names) != set(self.target_names):
            missing = set(self.source_names) - set(self.target_names)
            extra = set(self.target_names) - set(self.source_names)
            raise ValueError(
                f"[JointOrderMapper] Joint names do not match!\n"
                f"Missing in target: {missing}\n"
                f"Extra in target: {extra}"
            )

        # Precompute the index swaps so translation is instantaneous
        self.source_to_target_indices = np.array([
            self.source_names.index(name) for name in self.target_names
        ], dtype=np.int64)

        self.target_to_source_indices = np.array([
            self.target_names.index(name) for name in self.source_names
        ], dtype=np.int64)

    def to_target(self, canonical_array: np.ndarray) -> np.ndarray:
        """
        Takes a canonical MuJoCo array and rearranges it into the policy's custom order.
        """
        return canonical_array[..., self.source_to_target_indices]

    def to_source(self, policy_array: np.ndarray) -> np.ndarray:
        """
        Takes a policy's custom array and rearranges it back into standard MuJoCo order.
        """
        return policy_array[..., self.target_to_source_indices]