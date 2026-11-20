#!/usr/bin/env bash
# Standalone recording only; starts no planner, estimator or vehicle operation.
set -eo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$workspace/devel/setup.bash"
exec /usr/bin/python3 "$workspace/src/quaddock_logging/scripts/record_experiment.py" \
    --workspace "$workspace" "$@"
