#!/bin/bash

# 发布nav_msgs/Odometry消息到 /odom 话题
rostopic pub /target nav_msgs/Odometry "{
    header: {
        seq: 0,
        stamp: {
            secs: $(date +%s),
            nsecs: 0
        },
        frame_id: 'odom'
    },
    child_frame_id: 'base_link',
    pose: {
        pose: {
            position: {
                x: 5.0,
                y: 5.0,
                z: 1.0
            },
            orientation: {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0
            }
        },
        covariance: [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ]
    },
    twist: {
        twist: {
            linear: {
                x: 0.0,
                y: 0.0,
                z: 0.0
            },
            angular: {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        },
        covariance: [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ]
    }
}" -r 1
