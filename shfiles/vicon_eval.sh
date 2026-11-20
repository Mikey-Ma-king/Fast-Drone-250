#!/usr/bin/env bash
# Vicon receiver only. All output remains under /evaluation/vicon.
set -eo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$workspace/devel/setup.bash"
exec roslaunch mocap_vicon vicon.launch "$@"
