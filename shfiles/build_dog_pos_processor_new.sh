#!/usr/bin/env bash
# Build only the SW3 addition; do not rebuild the original processor/controllers.
set -eo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
source "$workspace/devel/setup.bash"
cmake -S "$workspace/src" -B "$workspace/build/sw3" -G 'Unix Makefiles' \
    '-DCATKIN_WHITELIST_PACKAGES=planning;quaddock_ours' \
    -DCATKIN_DEVEL_PREFIX="$workspace/devel" \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release
cmake --build "$workspace/build/sw3" --target dog_pos_processor_sw3 -j2
