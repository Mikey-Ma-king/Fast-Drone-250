#!/usr/bin/env bash
# Only the evaluation receiver and recording tools; no localization/control build.
set -eo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
source "$workspace/devel/setup.bash"
cmake -S "$workspace/src" -B "$workspace/build/vicon_logging" -G 'Unix Makefiles' \
    '-DCATKIN_WHITELIST_PACKAGES=mocap_base;mocap_vicon;quaddock_logging' \
    -DCATKIN_DEVEL_PREFIX="$workspace/devel" \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release
cmake --build "$workspace/build/vicon_logging" --target mocap_vicon_node -j2
