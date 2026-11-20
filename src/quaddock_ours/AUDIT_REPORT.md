# Deployment audit

Audit target: the real Fast-Drone-250 line (`planning/dog_pos_processor.cpp`), not the simulation target.

| Check | Result |
|---|---|
| Ours-SW3 core imports and finite output | PASS |
| Independent `uwb_range` and `uwb_bearing` events | PASS |
| Original AOA heartbeat and ±45° bearing gate represented | PASS in adapter |
| Finite range is not rejected by the bearing gate | PASS |
| `linktarck.py` changed | NO |
| Python unit tests | PASS (4 tests) |
| First-event split Jacobian replay | PASS (1x4 range/bearing rows) |
| Read-only real bag event replay (44--69 s) | PASS: 97 range + 97 bearing, zero estimator errors |
| Deployment parameters match frozen `sw3_joint_12` | PASS: 3 s / 4 / 0.5 / 0.5 |
| Catkin package build | PASS in a temporary ROS Noetic workspace |
| In-place modification of original `dog_pos_processor.cpp` | NOT APPLIED |
| Real hardware topic replay | NOT RUN |

The package is therefore a reviewed fusion adapter and handoff, not evidence that a real flight has already run. The receiving agent still needs to replace the original fusion block or make this package the sole `/dog_pos_processed` publisher, then replay the real topics before connecting the planner.

The source audit also found a pre-existing `simulate_mode_ = true` assignment in the original `dog_pos_processor.cpp`; confirm that setting before real flight because it zeros `camera_offset_`.
