# Evaluation: unitree_rl_gym_g1_motion

- checkpoint: `checkpoints/unitree_rl_gym_g1_motion.pt` (sha256 `cf668f75b90d1abf`)
- harness 1.0.0, physics 500 Hz, control 50 Hz, gains from **policy**
- seeds: [0, 1, 2], command (vx, vy, yaw_rate): [0.5, 0.0, 0.0]

Mean ± std over seeds. Velocity metrics exclude the first 1.0 s and are n/a if the robot fell before that.

| scenario | falls | survival | v_xy RMSE m/s | yaw RMSE rad/s | track RMSE rad | action rate rad/s | jerk rad/s^3 | power W | q-limit viol | tau sat |
|---|---|---|---|---|---|---|---|---|---|---|
| flat_nominal | 0% | 1.00 ± 0.00 | 0.114 ± 0.009 | 0.031 ± 0.004 | n/a | 5.93 ± 0.03 | 9576 ± 15 | 161 ± 0 | 0.0% ± 0.0% | 0.0% ± 0.0% |
| stand_still | 0% | 1.00 ± 0.00 | 0.095 ± 0.000 | 0.039 ± 0.000 | n/a | 5.59 ± 0.02 | 9024 ± 55 | 119 ± 1 | 0.0% ± 0.0% | 0.0% ± 0.0% |
| low_friction | 0% | 1.00 ± 0.00 | 0.065 ± 0.006 | 0.044 ± 0.003 | n/a | 5.69 ± 0.06 | 9067 ± 99 | 153 ± 2 | 0.0% ± 0.0% | 0.0% ± 0.0% |
| ice | 100% | 0.56 ± 0.21 | 0.313 ± 0.071 | 0.269 ± 0.038 | n/a | 7.24 ± 0.15 | 10419 ± 290 | 347 ± 14 | 5.8% ± 0.4% | 11.4% ± 1.7% |
| push_lateral_medium | 33% | 0.82 ± 0.26 | 0.250 ± 0.032 | 0.081 ± 0.030 | n/a | 5.59 ± 0.19 | 9084 ± 209 | 160 ± 9 | 3.6% ± 2.7% | 0.1% ± 0.1% |
| push_forward_hard | 33% | 0.86 ± 0.20 | 0.583 ± 0.079 | 0.157 ± 0.050 | n/a | 5.95 ± 0.09 | 9075 ± 141 | 203 ± 13 | 2.6% ± 3.6% | 4.9% ± 1.0% |
| push_backward_medium | 0% | 1.00 ± 0.00 | 0.174 ± 0.030 | 0.036 ± 0.002 | n/a | 5.58 ± 0.02 | 9114 ± 41 | 146 ± 1 | 0.0% ± 0.0% | 0.0% ± 0.0% |
