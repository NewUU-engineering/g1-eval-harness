# G1 Policy Evaluation Harness

One shared MuJoCo evaluation tool for every Unitree G1 policy.
Every policy runs on the **same robot model, physics, control rate, torque limits,
scenarios and seeds**, and gets the **same metrics, logs and videos**, so results
from different policies can be compared directly.

The harness owns everything that must stay identical. Each policy owner writes
one small **adapter** that translates between the harness and their policy.

```
g1-eval-harness/
├── evaluate.py                 # the one command
├── harness/                    # OWNED BY THE HARNESS - do not change per policy
│   ├── config.py               #   physics 500 Hz, control 50 Hz, fall rules, seeds
│   ├── robot.py                #   real G1 29-DOF model, name-based indexing, limits
│   ├── scenarios.py            #   flat, stand, low friction, ice, 3 pushes
│   ├── engine.py               #   sim loop, PD law, torque limits, pushes, logging
│   ├── metrics.py              #   all metrics (see METRICS.md)
│   ├── reporting.py            #   summary.json / summary.md / logs / videos
│   └── types.py                #   RobotState in, JointCommand out (LowState / LowCmd)
├── adapters/                   # ONE FILE PER POLICY
│   ├── base.py                 #   PolicyAdapter interface
│   ├── unitree_rl_gym_adapter.py   # working example (unitree_rl_gym pretrained G1)
│   ├── template_adapter.py     #   copy this to start your own
│   └── external/               #   policies that run in another process
│       ├── protocol.py         #     standalone TCP protocol (copy into your env)
│       ├── socket_bridge.py    #     harness side, lock-step, deterministic
│       └── unitree_dds_bridge.py #   harness acts as the robot over Unitree DDS
├── utils/
│   ├── joint_mapping.py        # Marvarid's joint-order mapper
│   ├── joint_orders.py         # known orders (IsaacLab, ...)
│   └── policy_runtime.py       # TorchScript loader (torch, or numpy fallback)
├── assets/g1/                  # Unitree's G1 model + meshes (BSD-3)
├── checkpoints/unitree_rl_gym_g1_motion.pt
├── examples/external_unitree_rl_gym_client.py
├── tests/
├── STATUS.md                   # deliverables + open items  <- start here
├── METRICS.md  DESIGN.md  FIXES.md
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.9+
pip install -r requirements.txt
pip install torch            # recommended; without it the numpy fallback runs the example policy
python -m pytest -q tests    # 15 tests, ~5 s
```

Videos are rendered offscreen. On Linux `evaluate.py` uses `MUJOCO_GL=egl`
(headless GPU); on a Linux machine without a GPU use `export MUJOCO_GL=osmesa` or `--no-video`.
On Windows and macOS nothing needs to be set. If you set `MUJOCO_GL` yourself on Windows,
remove it (`set MUJOCO_GL=`), because `egl`/`osmesa` are Linux-only.

**Windows `cmd`:** the multi-line commands below use `\` (Linux/macOS). In `cmd`, put the
command on one line or end lines with `^` instead.
Rendering is the slow part on CPU-only machines; the simulation itself runs
~20x faster than real time.

## 2. Evaluate a policy - one command

```bash
python evaluate.py \
  --adapter adapters.unitree_rl_gym_adapter:UnitreeRLGymAdapter \
  --checkpoint checkpoints/unitree_rl_gym_g1_motion.pt
```

Defaults: all 7 scenarios, seeds `0 1 2` (quick check; use 10 seeds for reports), command `vx=0.5 m/s`, gains from the policy.
Common options:

| option | meaning |
|---|---|
| `--scenarios flat_nominal ice` | subset (`--list-scenarios` shows all) |
| `--seeds 0 1 2 3 4` | more seeds = tighter confidence |
| `--cmd-vel 0.8 0 0` | velocity command `vx vy yaw_rate` (scenarios like `stand_still` override it) |
| `--gains harness` | force the shared G1 reference gains instead of each policy's own (see DESIGN.md) |
| `--set key=value` / `--adapter-config cfg.yaml` | pass options to your adapter |
| `--no-video`, `--video-all-seeds` | video control (default: first seed of each scenario) |
| `--output-dir results/x` | default `results/<policy>_<timestamp>` |

### What you get

```
results/<policy>_<timestamp>/
├── summary.md        # table: mean ± std over seeds per scenario  <- read this first
├── summary.json      # everything: config snapshot, per-trial metrics, aggregates,
│                     #   seed perturbations, checkpoint sha256, versions
├── logs/<scenario>_seed<k>.npz   # per-control-step time series (q, dq, targets, gains,
│                                  #   velocities, base pose, torques, latency)
└── videos/<scenario>_seed<k>.mp4
```

Expected result for the example policy (reference run in `examples/sample_results/`:
PyTorch backend, Windows 10, Python 3.10, MuJoCo 3.15, seeds 0 1 2):

| scenario | falls | v_xy RMSE m/s | power W |
|---|---|---|---|
| flat_nominal | 0% | 0.114 | 161 |
| stand_still | 0% | 0.095 | 119 |
| low_friction | 0% | 0.065 | 153 |
| ice | 100% | 0.313 | 347 |
| push_lateral_medium | 33% | 0.250 | 160 |
| push_forward_hard | 33% | 0.583 | 203 |
| push_backward_medium | 0% | 0.174 | 146 |

Videos for four scenarios are in `examples/sample_results/videos/`.

**Reproducibility.** The same policy + seed on the same machine and setup gives identical
results. Across machines, operating systems or backends (PyTorch vs the numpy fallback, CPU vs
GPU) tiny floating-point differences can change the outcome of trials where the robot is on the
edge of falling. Example: with the numpy backend, ice seed 0 survives the full 10 s; with
PyTorch it falls at 8.6 s, so the ice fall rate reads 67 % vs 100 %. All other scenarios match.
Therefore:

- **Compare policies on the same machine with the same environment.**
- **Use at least 10 seeds for any reported comparison** (`--seeds 0 1 2 3 4 5 6 7 8 9`);
  with 3 seeds one flipped trial moves a fall rate by 33 %.

If you get something very different from the table with the same seeds and the PyTorch
backend, your setup differs - check `summary.json → environment` and the checkpoint hash
(`cf668f75b90d1abf`).

## 3. Add your own policy

You never edit `harness/`. You write one adapter.

### 3a. Policy you can call from Python (most cases)

1. `cp adapters/template_adapter.py adapters/<yourname>_adapter.py`
2. Set `POLICY_JOINT_ORDER` to the joint order your network uses (29 names).
   Marvarid's mapper (`self.mapper`) converts to/from the harness's canonical order:
   `self.mapper.to_target(canonical) -> yours`, `self.mapper.to_source(yours) -> canonical`.
   Known orders are in `utils/joint_orders.py`; agree new ones with Marvarid.
3. Implement four methods:

```python
def default_pose(self):          # 29 canonical angles to spawn in
def reset(self, seed):           # clear RNN memory, last action, phase clock
def act(self, state):            # RobotState -> JointCommand(q, kp, kd[, dq, tau_ff])
def reference(self, t):          # optional: {"q": canonical[29]} for motion-tracking policies
```

`state` gives you `q, dq, tau, base_quat (wxyz), base_ang_vel (body), base_lin_acc (body),
projected_gravity ([0,0,-1] upright), command, t` - what the real robot measures.
`base_lin_vel` / `base_pos` are privileged; if you use them set
`uses_privileged_state = True` (it is recorded in the results).

4. Run it:

```bash
python evaluate.py --adapter adapters.<yourname>_adapter:YourAdapter --checkpoint path/to/policy.pt
```

5. Sanity-check before trusting numbers: watch `videos/flat_nominal_seed0.mp4`.
   If the robot collapses in the first second, it is almost always the adapter
   (joint order, default pose, gravity sign, scaling, gains) - see the checklist at
   the top of `template_adapter.py`, and compare with your training repo's own sim2sim.

### 3b. Policy in another process / environment (socket, deterministic)

If your policy lives in another venv, a ROS 2 node, or C++, copy
`adapters/external/protocol.py` there and subclass `PolicyClient`
(`examples/external_unitree_rl_gym_client.py` is a complete example).
The harness waits for your answer every control step, so results are
**bit-identical** to running in-process (verified by `tests/test_reference_policy.py`).

```bash
python evaluate.py --adapter adapters.external.socket_bridge:SocketBridgeAdapter --set port=7788
python examples/external_unitree_rl_gym_client.py --checkpoint checkpoints/unitree_rl_gym_g1_motion.pt --port 7788
```

(or let the harness launch the client: `--set "launch=python my_client.py --port 7788"`).

### 3c. Policy inside its own deploy stack over Unitree DDS (SONIC, BeyondMimic)

The harness pretends to be the robot: it publishes `rt/lowstate` from MuJoCo every
physics step and applies whatever arrives on `rt/lowcmd`, with the same PD law,
limits, scenarios and metrics. Simulation is paced to wall clock.

```bash
pip install unitree_sdk2py
python evaluate.py --adapter adapters.external.unitree_dds_bridge:UnitreeDDSBridgeAdapter \
    --set interface=lo --set domain_id=1 --scenarios flat_nominal --seeds 0
# then start the deploy stack on the same interface / domain id
```

> **Status:** written against the unitree_sdk2_python API but **not yet tested with a real
> SONIC / BeyondMimic deployment**. Expect to adjust `interface`, `domain_id`, `mode_machine`
> and the stack's start-up handshake. Real-time runs are not bit-reproducible; use more seeds.
> Where possible, prefer exporting the policy (ONNX/TorchScript) and using 3a - it is
> deterministic and much easier to debug.

## 4. Rules that keep it fair (owned by `harness/config.py`)

- Robot: Unitree `g1_29dof_rev_1_0` (assets/g1, generated by `build_harness_model.py`)
- Physics 500 Hz, policy 50 Hz; policies trained at another rate are rejected
- Torque limits from Unitree's model (88/139/50/25/5 N·m), commanded torque is clipped
- PD law `tau = kp(q* − q) + kd(dq* − dq) + tau_ff` computed by the harness every physics step
- Spawn on the ground with seeded ±0.02 rad joint noise; seeded push time/direction jitter
- Fall = pelvis < 0.40 m, tilt > 60°, or any body other than the feet touches the floor
- Any change to these goes through a reviewed commit to `harness/`, and bumps `HARNESS_VERSION`

Metric definitions: **METRICS.md**. Why it is designed this way: **DESIGN.md**.
What was wrong with the first version: **FIXES.md**. Status and open items: **STATUS.md**.
