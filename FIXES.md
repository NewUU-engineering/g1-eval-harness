# What was wrong in the first version, and what changed

v0 result: every policy fell within 0.3 s in every scenario (13–19 control steps).
v1 result: the unitree_rl_gym pretrained policy walks; fall rates now differ meaningfully
between scenarios (0 % flat, 33 % lateral and hard forward push, 100 % ice; 3 seeds).

## Root causes (each one alone made the robot fall)

1. **The robot model was not a G1.** `assets/g1_29dof.xml` was hand-built from boxes and
   cylinders with invented masses and lengths. The knee axis was reversed (feet ended up 23 cm
   in front of the pelvis), hip links started 5–6.5 cm inside the pelvis (8 penetrating contacts
   at t = 0), and the robot spawned 11 cm in the air. Even with no policy it collapsed in 0.44 s.
   → Replaced with Unitree's `g1_29dof_rev_1_0.xml` + meshes.
2. **Projected gravity had the wrong sign** (`[0,0,+1]` upright instead of `[0,0,-1]`): the
   network believed it was upside down from the first step.
   → Harness provides `projected_gravity` with the legged_gym convention.
3. **The checkpoint is an LSTM whose memory was never reset**, and the input-size "probe" ran
   the network on zeros, writing garbage into that memory. This is also why different seeds gave
   different numbers although the simulation had no randomness.
   → `reset()` calls `reset_memory()`; probing removed; obs size is fixed in the adapter.
4. **Adapter constants did not match unitree_rl_gym's `g1.yaml`**: command not scaled by
   `[2, 2, 0.25]`, default angles −0.2/0.4 instead of −0.1/0.3, gait period 0.64 s instead of 0.8 s.
5. **Uniform Kp = 60, Kd = 2** instead of the gains the policy was trained with
   (100/100/100/150/40/40, Kd 2/2/2/4/2/2).

## Harness problems fixed

| v0 | v1 |
|---|---|
| seeds did nothing (`np.random.seed` with no randomness) | seeded joint noise + push time/direction jitter |
| torque-limit violations always 0 (counted after clipping) | counted on commanded torque before clipping |
| one 88 N·m limit for every joint | per-joint limits from Unitree's model |
| velocity error compared world-frame velocity to a body-frame command | heading-frame velocity, 1 s window, warm-up excluded |
| tilt computed from the quaternion *before* stepping | fall checks on the post-step state, plus floor-contact check |
| `qpos[7:36]` hard-coded; engine reached into `adapter.mapper` | name-based indexing; `default_pose()` in the adapter interface |
| low friction set on all geoms but never restored | floor + robot geoms set per trial, restored afterwards |
| no motion-tracking error metric | `reference(t)` → `joint_pos_tracking_rmse` |
| SONIC adapter was a mock returning the current pose | working socket bridge (tested, bit-identical to in-process) + DDS bridge |
| joint mapper never exercised (all orders identical) | IsaacLab order + tests; external test sends IsaacLab order over the wire |
| README referenced a checkpoint that did not exist | README matches the repo; 15 automated tests |
