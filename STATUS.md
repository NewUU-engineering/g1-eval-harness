# Status and open items

## Deliverables

| deliverable | where | status |
|---|---|---|
| Evaluate a policy with one command | `evaluate.py` | done, verified on Windows 10 (PyTorch) and Linux (numpy backend) |
| Metric definitions (falls, tracking error, velocity tracking, smoothness, limit violations) | `METRICS.md` | done |
| Working example adapter | `adapters/unitree_rl_gym_adapter.py` (unitree_rl_gym pretrained G1) | done, walks in all non-ice scenarios |
| README: how to add your own policy | `README.md` §3 | done (in-process, socket, DDS) |
| Direct call **and** externally running policies, same metrics | `adapters/external/` | socket path done and tested; DDS path written, untested (below) |
| Design based on existing sim2sim setups | `DESIGN.md` | done |
| Tests | `tests/` (`python -m pytest -q tests`) | 22/22 pass (15 from v1.0.0 + 7 tracking) |
| **v2.0.0:** motion-tracking policies (motion-length episodes, failure against the reference, MPKPE / root-relative MPKPE / drift / strict success) | `harness/`, `METRICS.md` §3, `README.md` (top) | done; walking-policy results identical to v1.0.0 |
| **v2.0.0:** BeyondMimic adapter (mjlab ONNX export) | `adapters/beyondmimic_adapter.py` | done, checked against mjlab's evaluation on the LAFAN1 demo dance |
| **v2.0.0:** SONIC (ViT-H) adapter (ViT-H's fine-tune of SONIC, its `g1_tracker`, 1570-D deploy policy) | `adapters/sonic_adapter.py` | done, checked against `g1_tracker` on a tennis serve |
| **v2.0.0:** SONIC v1.1 adapter (NVIDIA's release, encoder + decoder, mode `g1`) | `adapters/sonic_v1_1_adapter.py` | done, checked against NVIDIA's sim-to-sim on a tennis serve, a jump kick and their squat |
| **v2.0.0:** robot-model choice, default `ros2` (ROS 2 stack's G1) | `assets/g1/build_ros2_model.py`, `--robot` | done; `--robot unitree_rl_gym` reproduces v1.0.0 |

Reference results: `examples/sample_results/` (summary, per-trial logs, videos; v1 robot model).
What was wrong with the first version and what changed: `FIXES.md`.

## Open items

1. **DDS bridge not yet tested with real stacks.** `adapters/external/unitree_dds_bridge.py`
   (for SONIC's C++ deploy and BeyondMimic's ROS 2 controller) follows the unitree_sdk2py API and
   the unitree_mujoco pattern, but needs one session with each policy owner to confirm network
   interface, domain id, `mode_machine` and the stack's start-up handshake. Both policies can now be
   evaluated in-process instead (v2.0.0 adapters).
2. **IsaacLab joint order: resolved for BeyondMimic (v2.0.0).** An mjlab BeyondMimic export stores
   `joint_names` in its ONNX metadata, and it is the canonical MuJoCo order (not IsaacLab's). SONIC
   (`g1_tracker`) uses IsaacLab order inside the network and MuJoCo order outside; its adapter reuses
   g1_tracker's own mapping. Original note: `utils/joint_orders.py` lists the order IsaacLab-trained
   policies (e.g. BeyondMimic) are expected to use. It should be checked against the joint names
   stored with an actual BeyondMimic export. The joint-mapping script (`utils/joint_mapping.py`)
   is used unchanged.
3. **PD gains - decision needed.** Default: each policy sends the gains it was trained with
   (as on the real robot). `--gains harness` forces one shared gain set instead. Rationale in
   `DESIGN.md` §4. Both are supported; only the default needs agreeing on. The v2.0.0 adapters
   use the default (their own gains).
4. **Comparison protocol.** For reported comparisons: same machine and environment for all
   policies, 10 seeds (`--seeds 0 1 2 3 4 5 6 7 8 9`). Trials on the edge of falling can flip
   across machines/backends (see README, Reproducibility).
5. **Motion trackers and pushes (v2.0.0).** Pushes start at about 4 s, so motions shorter than
   about 4.3 s are never pushed. `stand_still` is the same as `flat_nominal` for motion trackers
   (they have no velocity command).
6. **Privileged state in the tracking adapters (v2.0.0).** BeyondMimic (anchor position, base linear
   velocity) and SONIC (ViT-H) (base position) read the true base state; on the robot it comes from a
   state estimator. Both adapters set `uses_privileged_state = True`; SONIC v1.1 doesn't need it. An estimator-noise scenario would
   make this closer to the robot.
7. **Starting state of motion trackers (v2.0.0).** The robot starts at rest on the reference's first
   frame (plus the seeded joint noise). mjlab and `g1_tracker` also start it with the reference
   velocities, which matters for motions that begin while moving.

8. **Default robot model (v2.0.0) - open decision.** Results change with the model: on
   `ros2` the example walking policy tracks speed better (0.03 vs 0.11 m/s) and survives ice, but
   falls more under a sideways push; BeyondMimic finishes the LAFAN1 cartwheel only on `ros2`.
   `examples/sample_results/` are still v1 numbers.

## Possible extensions

Terrain scenarios (slopes, steps, rough ground), observation latency/noise scenarios,
joint-velocity limit checks, a leaderboard script that merges several `summary.json` files.
