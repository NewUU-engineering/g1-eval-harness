"""
harness/engine.py

The one simulation loop every policy goes through.
The harness owns: the robot model, physics timestep, control rate, the PD law and
torque limits, spawning, disturbances, fall detection, logging and metrics.
The adapter only maps RobotState -> JointCommand.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mujoco
import numpy as np

from harness.config import HarnessConfig, G1_REFERENCE_GAINS
from harness.robot import RobotModel
from harness.scenarios import Scenario
from harness.metrics import TrialMetrics, compute_metrics
from harness.types import RobotState, JointCommand


@dataclass
class TrialResult:
    scenario: str
    seed: int
    metrics: TrialMetrics
    log: Dict[str, np.ndarray]
    frames: List[np.ndarray] = field(default_factory=list)
    perturbations: Dict = field(default_factory=dict)


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def projected_gravity(R_world_body: np.ndarray) -> np.ndarray:
    """Unit gravity direction expressed in the body frame ([0,0,-1] when upright)."""
    return R_world_body.T @ np.array([0.0, 0.0, -1.0])


class SimulationEngine:
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.robot = RobotModel(cfg.model_xml, cfg.physics_dt)
        self.model = self.robot.model
        self.allowed_contact_bodies = {self.robot._id(mujoco.mjtObj.mjOBJ_BODY, b)
                                       for b in cfg.allowed_ground_contact_bodies}
        if cfg.gains_mode not in ("policy", "harness"):
            raise ValueError("gains_mode must be 'policy' or 'harness'")
        self.ref_kp = np.array([G1_REFERENCE_GAINS[n][0] for n in self.robot.joint_names])
        self.ref_kd = np.array([G1_REFERENCE_GAINS[n][1] for n in self.robot.joint_names])

    # ------------------------------------------------------------------ helpers
    def _state(self, d: mujoco.MjData, t0: float, command: np.ndarray, last_tau: np.ndarray) -> RobotState:
        r = self.robot
        R = r.base_rot(d)
        if r.pelvis_acc_sensor is not None:
            adr, dim = r.pelvis_acc_sensor
            acc = d.sensordata[adr:adr + dim].copy()
        else:
            acc = R.T @ np.array([0.0, 0.0, 9.81])
        return RobotState(
            t=float(d.time - t0),
            q=r.q(d), dq=r.dq(d), tau=last_tau.copy(),
            base_quat=r.base_quat(d),
            base_ang_vel=r.base_ang_vel_body(d),
            base_lin_acc=acc,
            projected_gravity=projected_gravity(R),
            command=command.copy(),
            base_pos=r.base_pos(d),
            base_lin_vel=R.T @ r.base_lin_vel_world(d),
        )

    def _check_fall(self, d: mujoco.MjData) -> str:
        r, c = self.robot, self.cfg
        if r.base_pos(d)[2] < c.fall_min_base_height_m:
            return "base_too_low"
        tilt = float(np.arccos(np.clip(r.base_rot(d)[2, 2], -1.0, 1.0)))
        if tilt > c.fall_max_tilt_rad:
            return "base_tilt"
        for b in r.ground_contact_bodies(d):
            if b not in self.allowed_contact_bodies:
                return f"ground_contact:{r.body_name(b)}"
        return ""

    # ------------------------------------------------------------------ trial
    def run_trial(self, adapter, scenario: Scenario, seed: int, command: np.ndarray,
                  record_video: bool = False) -> TrialResult:
        cfg, r, m = self.cfg, self.robot, self.model
        if adapter.command_rate not in ("control", "physics"):
            raise ValueError("adapter.command_rate must be 'control' or 'physics'")
        if adapter.command_rate == "control" and abs(adapter.policy_dt - cfg.control_dt) > 1e-9:
            raise ValueError(f"Policy expects dt={adapter.policy_dt}s but the harness runs at "
                             f"dt={cfg.control_dt}s. The harness control rate is fixed for fairness.")

        rng = np.random.default_rng(seed)
        command = np.asarray(scenario.command_override if scenario.command_override is not None else command,
                             dtype=np.float64)

        # ---- seeded perturbations (identical for every policy at this seed) ----
        pushes = []
        for p in scenario.pushes:
            dt_jit = rng.uniform(-cfg.push_time_jitter_s, cfg.push_time_jitter_s)
            ang = np.deg2rad(rng.uniform(-cfg.push_dir_jitter_deg, cfg.push_dir_jitter_deg))
            Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
            pushes.append({"start": p.time_s + dt_jit, "end": p.time_s + dt_jit + p.duration_s,
                           "force": Rz @ np.asarray(p.force_xyz, dtype=float)})
        joint_noise = rng.uniform(-cfg.init_joint_noise_rad, cfg.init_joint_noise_rad, size=29)

        # ---- reset world ----
        r.set_ground_friction(scenario.friction)
        d = mujoco.MjData(m)
        adapter.reset(seed)
        q0 = np.asarray(adapter.default_pose(), dtype=np.float64).reshape(-1)
        if q0.shape != (29,):
            raise ValueError("adapter.default_pose() must return 29 canonical joint angles.")
        q0 = np.clip(q0 + joint_noise, r.q_lower, r.q_upper)
        d.qpos[r.base_qpos_adr:r.base_qpos_adr + 3] = [0.0, 0.0, 1.0]
        d.qpos[r.base_qpos_adr + 3:r.base_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[r.qpos_adr] = q0
        r.place_on_ground(d, cfg.spawn_clearance_m)
        t0 = float(d.time)

        renderer, cam = None, None
        if record_video:
            try:
                renderer = mujoco.Renderer(m, height=cfg.video_height, width=cfg.video_width)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = r.base_body
                cam.distance, cam.azimuth, cam.elevation = 3.0, 135.0, -15.0
            except Exception as e:  # no OpenGL available: keep evaluating, skip video
                print(f"    [VIDEO] Offscreen rendering unavailable ({e}); continuing without video.")
                renderer = None

        n_ctrl = int(round(scenario.duration_s / cfg.control_dt))
        rows: Dict[str, list] = {k: [] for k in ("t", "q", "dq", "q_target", "kp", "kd", "cmd", "vel_heading",
                                               "yaw_rate", "base_pos", "base_quat", "tau", "latency_ms",
                                               "q_ref", "q_ref_mask")}
        phys = {"steps": 0, "power_sum": 0.0, "tau_sq_sum": 0.0, "sat_steps": 0, "sat_joint_steps": 0, "sim_time": 0.0}
        fall_reason, fall_time = "", None
        last_tau = np.zeros(29)
        frames: List[np.ndarray] = []
        cmd_obj: Optional[JointCommand] = None

        def get_command(state: RobotState) -> (JointCommand, float):
            tic = time.perf_counter()
            jc = adapter.act(state)
            lat = (time.perf_counter() - tic) * 1e3
            jc = jc.validated()
            if cfg.gains_mode == "harness":
                jc.kp, jc.kd = self.ref_kp.copy(), self.ref_kd.copy()
            return jc, lat

        for k in range(n_ctrl):
            state = self._state(d, t0, command, last_tau)
            cmd_obj, latency = get_command(state)
            if not cmd_obj.is_finite():
                fall_reason, fall_time = "invalid_action", state.t
                break
            ref = adapter.reference(state.t)

            for s in range(cfg.decimation):
                if adapter.command_rate == "physics" and s > 0:
                    cmd_obj, _ = get_command(self._state(d, t0, command, last_tau))
                    if not cmd_obj.is_finite():
                        fall_reason = "invalid_action"
                        break
                t_now = d.time - t0
                d.xfrc_applied[r.base_body, :] = 0.0
                for p in pushes:
                    if p["start"] <= t_now < p["end"]:
                        d.xfrc_applied[r.base_body, :3] += p["force"]
                q, dq = d.qpos[r.qpos_adr], d.qvel[r.qvel_adr]
                tau_cmd = cmd_obj.kp * (cmd_obj.q - q) + cmd_obj.kd * (cmd_obj.dq - dq) + cmd_obj.tau_ff
                sat = np.abs(tau_cmd) > r.torque_limit
                tau = np.clip(tau_cmd, -r.torque_limit, r.torque_limit)
                d.ctrl[r.actuator_ids] = tau
                warn_before = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
                mujoco.mj_step(m, d)
                if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > warn_before:
                    fall_reason = "sim_unstable"
                    break
                last_tau = tau
                phys["steps"] += 1
                phys["power_sum"] += float(np.sum(np.abs(tau * d.qvel[r.qvel_adr])))
                phys["tau_sq_sum"] += float(np.sum(tau * tau))
                phys["sat_steps"] += int(sat.any())
                phys["sat_joint_steps"] += int(sat.sum())
            phys["sim_time"] = float(d.time - t0)

            # ---- log this control step (post-physics state) ----
            yaw = quat_to_yaw(r.base_quat(d))
            v_w = r.base_lin_vel_world(d)
            c, s_ = np.cos(-yaw), np.sin(-yaw)
            v_h = np.array([c * v_w[0] - s_ * v_w[1], s_ * v_w[0] + c * v_w[1], v_w[2]])
            yaw_rate = float((r.base_rot(d) @ r.base_ang_vel_body(d))[2])
            rows["t"].append(state.t)
            rows["q"].append(r.q(d)); rows["dq"].append(r.dq(d))
            rows["q_target"].append(cmd_obj.q.copy()); rows["kp"].append(cmd_obj.kp.copy()); rows["kd"].append(cmd_obj.kd.copy())
            rows["cmd"].append(command.copy()); rows["vel_heading"].append(v_h); rows["yaw_rate"].append(yaw_rate)
            rows["base_pos"].append(r.base_pos(d)); rows["base_quat"].append(r.base_quat(d))
            rows["tau"].append(last_tau.copy()); rows["latency_ms"].append(latency)
            if ref is not None and "q" in ref:
                rows["q_ref"].append(np.asarray(ref["q"], dtype=float))
                rows["q_ref_mask"].append(np.asarray(ref.get("q_mask", np.ones(29, bool)), dtype=bool))

            if renderer is not None and k % cfg.video_stride == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

            if fall_reason:
                fall_time = phys["sim_time"]
                break
            fall_reason = self._check_fall(d)
            if fall_reason:
                fall_time = phys["sim_time"]
                break

        if renderer is not None:
            renderer.close()
        r.restore_nominal()

        log = {k: np.asarray(v) for k, v in rows.items() if len(v)}
        if "q_ref" in log and len(log["q_ref"]) != len(log["t"]):
            del log["q_ref"], log["q_ref_mask"]  # reference must be given every step or not at all
        metrics = compute_metrics(
            log, control_dt=cfg.control_dt, duration_s=scenario.duration_s, warmup_s=cfg.tracking_warmup_s, window_s=cfg.tracking_window_s,
            q_lower=r.q_lower, q_upper=r.q_upper, soft_fraction=cfg.soft_joint_limit_fraction,
            fall_reason=fall_reason, fall_time=fall_time, phys=phys)
        perturb = {"command": command.tolist(), "friction": scenario.friction,
                   "pushes": [{"start_s": p["start"], "duration_s": p["end"] - p["start"],
                               "force_xyz": p["force"].tolist()} for p in pushes],
                   "initial_joint_noise": joint_noise.tolist()}
        return TrialResult(scenario.name, seed, metrics, log, frames, perturb)
