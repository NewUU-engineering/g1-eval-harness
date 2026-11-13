# Design

## 1. How existing projects run sim2sim

| project | how the policy is run in MuJoCo | consequence for us |
|---|---|---|
| **unitree_rl_gym** (`deploy/deploy_mujoco`) | One Python script loads a TorchScript policy, builds the observation itself, runs the policy at 50 Hz and a PD loop at 500 Hz on Unitree's G1 MJCF. Gains, default angles, scales and the 0.8 s gait period live in a per-robot YAML (`configs/g1.yaml`). Verified: our adapter reproduces it. | Direct-call policies. Observation and gains are policy-specific: they must live in the adapter. |
| **unitree_mujoco** | Runs MuJoCo as a stand-in for the robot: publishes `rt/lowstate` and consumes `rt/lowcmd` over DDS, so the *same* binary that drives the real robot can drive the sim. | Pattern for evaluating stacks we cannot (or should not) call directly. |
| **BeyondMimic** (whole_body_tracking) | Trained in IsaacLab; policy exported (ONNX) together with its deployment parameters. Deployed by a ROS 2 controller that also runs the reference motion. Joint order is IsaacLab's, not Unitree's. *(Verify exact export/metadata details with the BeyondMimic owner.)* | Needs joint remapping. Best evaluated by loading the exported policy in an adapter (deterministic) and providing `reference(t)` for tracking error; the ROS 2 path works through the DDS bridge if the controller is pointed at the Unitree SDK interface. |
| **SONIC** | C++ deployment stack communicating with the robot over Unitree DDS; sim2sim runs it against a MuJoCo process that behaves like the robot. *(Verify with the SONIC owner which topics / handshake it expects.)* | Exactly the unitree_mujoco pattern → DDS bridge. |

Common lesson: **sim2sim failures are almost always mismatches between training and
deployment** (joint order, default pose, observation scaling/sign, gains, control rate,
robot model). The first version of this harness had all of these (see FIXES.md). The
design below tries to make each of them either impossible or loudly visible.

## 2. Split of responsibilities

| harness owns (identical for everyone) | adapter owns (per policy) |
|---|---|
| robot model, physics 500 Hz, control 50 Hz | observation layout, scaling, history |
| PD law and torque limits, clipping | network loading and inference, RNN state |
| spawning, scenarios, seeded disturbances | joint order (via Marvarid's mapper) |
| fall detection, metrics, logs, videos | default pose, action scaling |
| result format, config snapshot | gains the policy was trained with |
| | reference motion (tracking policies) |

The interface between them mirrors Unitree's LowState / LowCmd:
`RobotState` (q, dq, tau, IMU quaternion/gyro/accelerometer, command, t) in,
`JointCommand` (q, dq, kp, kd, tau_ff per joint) out. Because it is the same information
the real robot exchanges, an in-process adapter, a socket client and a DDS deploy stack are
three transports for the same thing, and the metrics code never knows which one was used.

## 3. Three ways to run a policy

1. **In-process adapter** (`adapters/*.py`) - deterministic, fastest, easiest to debug. Default.
2. **Socket bridge** (`adapters/external/socket_bridge.py`) - policy in another process
   (another venv, ROS 2 Python node, C++). Lock-step: sim time stops until the command
   arrives, so results are bit-identical to (1). Tested.
3. **DDS bridge** (`adapters/external/unitree_dds_bridge.py`) - harness acts as the robot for an
   unmodified deploy stack. Wall-clock paced, so not bit-reproducible; latency/staleness stats
   are saved. Untested against real stacks at the time of writing.

Recommendation: use (1) or (2) for comparisons; use (3) to check that the real deploy stack
behaves the same as the exported policy.

## 4. PD gains - a decision for the team

The task says "same PD gains". On the real G1, however, **gains are sent by the policy in every
LowCmd**; unitree_rl_gym, BeyondMimic and SONIC were each trained with different gains. Forcing
one gain set on all of them runs every policy outside the conditions it was trained for, so the
comparison would measure gain mismatch, not policy quality.

Implemented: `--gains policy` (default) - each adapter sends its training gains, the harness
applies them with the same PD law and torque limits and records them in the log.
`--gains harness` forces the shared table in `harness/config.py` for anyone who wants the strict
reading. Both are one flag apart; the choice is recorded in `summary.json`.

## 5. Fairness and reproducibility details

- **Robot model**: Unitree's own `g1_29dof_rev_1_0.xml`; the only change (joint armature /
  damping / friction loss, matching unitree_rl_gym's sim2sim model) is applied by a script, so
  the difference from upstream is auditable.
- **Name-based indexing everywhere** (`harness/robot.py`): a wrong model or joint order fails at
  start-up instead of silently scrambling joints.
- **Seeds**: per-trial `numpy.random.default_rng(seed)` draws initial joint noise, push time and
  push direction. Same seed → same disturbances for every policy; same seed + same policy on the same
  machine/backend → bit-identical trial (tested). Across machines or backends, trials on the edge
  of falling can flip (see README, Reproducibility), so comparisons are run on one machine.
- **Friction scenarios** set friction on the floor *and* the robot's geoms, because MuJoCo takes
  the maximum of the two.
- **Torque saturation** is counted on the commanded torque before clipping.
- **Policy rate** must match the harness rate; a policy trained at 100 Hz is rejected rather than
  silently run at 50 Hz.
- **Results** store the harness version, full config, scenario definitions, actual seed
  perturbations, checkpoint SHA-256 and library versions.

## 6. Known limitations / next steps

- No terrain scenarios yet (slopes, steps, rough ground). Adding one = a new MJCF scene + a
  `Scenario` entry.
- No sensor noise or latency injection (deliberately: the harness measures the policy, not the
  adapter's noise model). A `latency_steps` scenario option would be a useful addition.
- Joint velocity limits are not checked (not in Unitree's MJCF).
- The DDS bridge needs a first real run with SONIC and BeyondMimic owners.
- `ISAACLAB_G1_29DOF_ORDER` should be confirmed against BeyondMimic's exported metadata (Marvarid).
