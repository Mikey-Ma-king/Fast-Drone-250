#!/usr/bin/env bash
set -eo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$workspace/devel/setup.bash"
exec /usr/bin/python3 "$workspace/src/quaddock_logging/scripts/mark_event.py" "$@"
