#!/usr/bin/env bash
set -eo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$workspace/devel/setup.bash"

planner_pid=""
recorder_pid=""
cleanup() {
    # Finalize rosbag before stopping the planner launch owned by this script.
    if [[ -n "$recorder_pid" ]] && kill -0 "$recorder_pid" 2>/dev/null; then
        kill -TERM "$recorder_pid"
        wait "$recorder_pid" || true
    fi
    if [[ -n "$planner_pid" ]] && kill -0 "$planner_pid" 2>/dev/null; then
        kill -INT "$planner_pid"
        wait "$planner_pid" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Run concurrently: foreground roslaunch used to block recording indefinitely.
roslaunch planning traj_server.launch &
planner_pid=$!
/usr/bin/python3 "$workspace/src/quaddock_logging/scripts/record_experiment.py" \
    --workspace "$workspace" "$@" &
recorder_pid=$!
wait -n "$planner_pid" "$recorder_pid"
