"""
harness/robot.py

Loads the canonical G1 MJCF and resolves everything BY NAME
(no hard-coded qpos slices), so a wrong model or joint order fails loudly.
"""
from typing import Dict, List
import numpy as np
import mujoco

from utils.joint_mapping import CANONICAL_G1_JOINTS


class RobotModel:
    def __init__(self, model_xml: str, physics_dt: float):
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.model.opt.timestep = physics_dt
        m = self.model

        self.joint_names: List[str] = list(CANONICAL_G1_JOINTS)
        self.joint_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names])
        self.qpos_adr = m.jnt_qposadr[self.joint_ids].copy()
        self.qvel_adr = m.jnt_dofadr[self.joint_ids].copy()

        # One torque actuator per joint, looked up by the joint it drives.
        act_for_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        missing = [n for n, j in zip(self.joint_names, self.joint_ids) if int(j) not in act_for_joint]
        if missing:
            raise ValueError(f"Model has no actuator for joints: {missing}")
        self.actuator_ids = np.array([act_for_joint[int(j)] for j in self.joint_ids])

        # Floating base
        self.base_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError("Expected exactly one free joint (floating base).")
        self.base_qpos_adr = int(m.jnt_qposadr[free[0]])
        self.base_qvel_adr = int(m.jnt_dofadr[free[0]])

        # Limits (from Unitree's model, not invented)
        self.q_lower = m.jnt_range[self.joint_ids, 0].copy()
        self.q_upper = m.jnt_range[self.joint_ids, 1].copy()
        self.torque_limit = m.jnt_actfrcrange[self.joint_ids, 1].copy()
        if np.any(self.torque_limit <= 0):
            bad = [n for n, t in zip(self.joint_names, self.torque_limit) if t <= 0]
            raise ValueError(f"Joints without actuatorfrcrange torque limits: {bad}")

        # Floor + bodies allowed to touch it
        self.floor_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.robot_geoms = np.array([g for g in range(m.ngeom)
                                     if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])])
        self._nominal_friction = m.geom_friction.copy()

        # Accelerometer for a realistic IMU reading, if the model has one at the pelvis.
        self.pelvis_acc_sensor = None
        for s in range(m.nsensor):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) or ""
            if m.sensor_type[s] == mujoco.mjtSensor.mjSENS_ACCELEROMETER and "pelvis" in name:
                self.pelvis_acc_sensor = (int(m.sensor_adr[s]), int(m.sensor_dim[s]))
        # Sensor noise must be OFF so results are deterministic (noise is the adapter's business).
        m.sensor_noise[:] = 0.0

    def _id(self, objtype, name: str) -> int:
        i = mujoco.mj_name2id(self.model, objtype, name)
        if i < 0:
            raise ValueError(f"'{name}' not found in robot model. Is this the canonical G1 29-DOF model?")
        return i

    # ---------- per-trial model edits ----------
    def set_ground_friction(self, mu: float):
        """Set sliding friction on the floor AND every robot collision geom.
        MuJoCo uses the max of the two geoms' friction, so setting only the floor would not lower it."""
        self.model.geom_friction[:] = self._nominal_friction
        self.model.geom_friction[self.floor_geom, 0] = mu
        self.model.geom_friction[self.robot_geoms, 0] = mu

    def restore_nominal(self):
        self.model.geom_friction[:] = self._nominal_friction

    # ---------- state access ----------
    def q(self, d: mujoco.MjData) -> np.ndarray:
        return d.qpos[self.qpos_adr].copy()

    def dq(self, d: mujoco.MjData) -> np.ndarray:
        return d.qvel[self.qvel_adr].copy()

    def base_pos(self, d) -> np.ndarray:
        return d.qpos[self.base_qpos_adr:self.base_qpos_adr + 3].copy()

    def base_quat(self, d) -> np.ndarray:
        return d.qpos[self.base_qpos_adr + 3:self.base_qpos_adr + 7].copy()

    def base_lin_vel_world(self, d) -> np.ndarray:
        return d.qvel[self.base_qvel_adr:self.base_qvel_adr + 3].copy()

    def base_ang_vel_body(self, d) -> np.ndarray:
        # For a free joint MuJoCo stores angular velocity in the body frame.
        return d.qvel[self.base_qvel_adr + 3:self.base_qvel_adr + 6].copy()

    def base_rot(self, d) -> np.ndarray:
        return d.xmat[self.base_body].reshape(3, 3).copy()

    def ground_contact_bodies(self, d) -> List[int]:
        bodies = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.floor_geom:
                bodies.append(int(self.model.geom_bodyid[c.geom2]))
            elif c.geom2 == self.floor_geom:
                bodies.append(int(self.model.geom_bodyid[c.geom1]))
        return bodies

    def body_name(self, b: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)

    def place_on_ground(self, d: mujoco.MjData, clearance: float):
        """Shift the base vertically so the lowest robot collision geom is `clearance` above the floor."""
        mujoco.mj_forward(self.model, d)
        fromto = np.zeros(6)
        dmin = np.inf
        for g in self.robot_geoms:
            dist = mujoco.mj_geomDistance(self.model, d, int(g), int(self.floor_geom), 5.0, fromto)
            dmin = min(dmin, dist)
        d.qpos[self.base_qpos_adr + 2] += clearance - dmin
        mujoco.mj_forward(self.model, d)

    def describe(self) -> Dict:
        return {
            "nq": int(self.model.nq), "nv": int(self.model.nv), "nu": int(self.model.nu),
            "total_mass_kg": float(self.model.body_subtreemass[self.base_body]),
            "torque_limits": dict(zip(self.joint_names, self.torque_limit.tolist())),
        }
