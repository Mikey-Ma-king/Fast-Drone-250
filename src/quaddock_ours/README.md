# `quaddock_ours`: Fast-Drone-250 deployment package

This directory is the handoff package for the frozen **Ours-SW3** estimator on the real Fast-Drone-250 line. It is a small catkin package that can be copied into the receiving machine's ROS workspace; it contains only the deployable estimator, its ROS adapter, configuration, and launch file. It is not a simulation node.

The node subscribes to the message shapes already used by Fast-Drone-250:

| Role | Default topic | Message and decoding |
|---|---|---|
| UAV odometry | `/vins_fusion/imu_propagate` | `nav_msgs/Odometry`, normal quaternion, world position/velocity |
| Dog odometry | `/dog_pos` | `nav_msgs/Odometry`, position/velocity; yaw is `pose.orientation.w` |
| Relative vision frontend | `/target_ekf_odom` | `nav_msgs/Odometry`, world target position; yaw is `pose.orientation.w` |
| UWB/AOA | `/AOA_Tag_data` | `pose.position.x=range`, `pose.orientation.x=bearing` |

The output `/quaddock/target_odom` is a normal world-frame `nav_msgs/Odometry` estimate. The optional `/quaddock/legacy_preview` is diagnostic only; the original `dog_pos_processor` must remain the sole publisher of `/dog_pos_processed` until the fusion block is integrated in place. The estimator status is published as JSON on `/quaddock/ours_sw3/status`.

AOA is split at this upper layer into two estimator events. Every finite received distance is sent as `uwb_range` whenever available. The angle is sent separately as `uwb_bearing` only when the mirrored original AOA/VINS availability state and ±45° heading gate allow it. The range is not discarded when that angle gate rejects the bearing. `linktarck.py` and its filtering are untouched.

`vision_input_mode` defaults to `world_odom` because the old `target_ekf_odom` producer publishes a world-frame target position. If the receiving agent replaces that frontend with an UAV-body relative vector, change it to `relative_odom` before testing. All topic names and calibration/lever-arm values are in `config/ours_real.yaml`.

The estimator values in that file are the frozen `sw3_joint_12` configuration: 3 s window, process scale 4, yaw process ratio 0.5, and translation curvature scale 0.5.

The package only estimates and publishes state. It never arms, takes off, lands, or publishes `PositionCommand`. The receiving agent must integrate `/quaddock/target_odom` into the original fusion block, or deliberately make this package the sole publisher after preserving the original readiness fields. Do not run two `/dog_pos_processed` publishers.

The exact scope and remaining verification limits are recorded in [AUDIT_REPORT.md](AUDIT_REPORT.md). The original-line integration boundary is in [ORIGINAL_FASTDRONE250_INTEGRATION.md](ORIGINAL_FASTDRONE250_INTEGRATION.md).

## Copy and build

Copy this directory into the Fast-Drone-250 catkin source space, then build it from a clean machine-local workspace:

```bash
cd ~/Fast-Drone-250
catkin_make --pkg quaddock_ours
source devel/setup.bash
roslaunch quaddock_ours ours_real.launch
```

The node requires Python 3 with NumPy and SciPy on the receiving machine. Do not copy `build/`, `devel/`, or precompiled extensions between machines. The default launch has `respawn=false` so a configuration or dependency failure is visible instead of silently restarting.

## Checks before connecting the planner

1. Start an isolated ROS master and replay representative real-line messages without a vehicle connected.
2. Confirm the dog and target yaw fields against the producer revision; set `dog_yaw_mode` or `vision_yaw_mode` to `quaternion` if those topics are changed to standard quaternions.
3. Confirm `target_ekf_odom` is world-frame before using `world_odom`; otherwise use `relative_odom`.
4. Verify the output frame, lever arms, timestamps, and `sensor_ages` in the status/log stream.
5. Check that stale or unmatched inputs remain visible as `unmatched`/`rejected` counters and that no control topic is advertised by this package.

The included `test/test_core.py` checks the copied estimator without ROS:

```bash
PYTHONPATH=deployment/quaddock_ours/src python3 -m unittest discover -s deployment/quaddock_ours/test
```
