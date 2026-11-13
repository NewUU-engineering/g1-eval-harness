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
| Tests | `tests/` (`python -m pytest -q tests`) | 15/15 pass |

Reference results: `examples/sample_results/` (summary, per-trial logs, videos).
What was wrong with the first version and what changed: `FIXES.md`.

## Open items

1. **DDS bridge not yet tested with real stacks.** `adapters/external/unitree_dds_bridge.py`
   (for SONIC's C++ deploy and BeyondMimic's ROS 2 controller) follows the unitree_sdk2py API and
   the unitree_mujoco pattern, but needs one session with each policy owner to confirm network
   interface, domain id, `mode_machine` and the stack's start-up handshake.
2. **IsaacLab joint order unverified.** `utils/joint_orders.py` lists the order IsaacLab-trained
   policies (e.g. BeyondMimic) are expected to use. It should be checked against the joint names
   stored with an actual BeyondMimic export. The joint-mapping script (`utils/joint_mapping.py`)
   is used unchanged.
3. **PD gains - decision needed.** Default: each policy sends the gains it was trained with
   (as on the real robot). `--gains harness` forces one shared gain set instead. Rationale in
   `DESIGN.md` §4. Both are supported; only the default needs agreeing on.
4. **Comparison protocol.** For reported comparisons: same machine and environment for all
   policies, 10 seeds (`--seeds 0 1 2 3 4 5 6 7 8 9`). Trials on the edge of falling can flip
   across machines/backends (see README, Reproducibility).

## Possible extensions

Terrain scenarios (slopes, steps, rough ground), observation latency/noise scenarios,
joint-velocity limit checks, a leaderboard script that merges several `summary.json` files.
