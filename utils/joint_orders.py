"""
utils/joint_orders.py

Joint orders used by the policies we evaluate. Feed these to Marvarid's
JointOrderMapper (utils/joint_mapping.py).

  CANONICAL_G1_JOINTS / UNITREE_DDS_ORDER   (in utils/joint_mapping.py)
      MuJoCo kinematic-tree order of Unitree's G1 model = Unitree motor index order
      on rt/lowstate / rt/lowcmd. Used by unitree_rl_gym and by DDS deploy stacks (SONIC).

  ISAACLAB_G1_29DOF_ORDER
      Breadth-first order IsaacLab produces when it parses the G1 USD/URDF
      (left/right/waist interleaved). Policies trained in IsaacLab (e.g. BeyondMimic)
      output actions in this order.
      >>> VERIFY against the policy's own config/ONNX metadata before trusting it
          (BeyondMimic exports the joint names in the ONNX metadata). Coordinate with Marvarid.
"""
ISAACLAB_G1_29DOF_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
