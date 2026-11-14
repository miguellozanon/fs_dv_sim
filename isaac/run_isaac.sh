#!/usr/bin/env bash
# Launch the Isaac Sim backend on native Linux with Isaac's bundled Python.
#   ISAAC_SIM=~/isaacsim ./isaac/run_isaac.sh --mission trackdrive [--headless]
# The ROS 2 relay may run on the same machine (default 127.0.0.1) or elsewhere (--relay-host).
# On Windows use isaac/run_isaac.ps1 instead.
set -euo pipefail
: "${ISAAC_SIM:?set ISAAC_SIM to your Isaac Sim install folder (the one containing python.sh)}"
cd "$(dirname "$0")/.."
"$ISAAC_SIM/python.sh" -c "import yaml" 2>/dev/null || "$ISAAC_SIM/python.sh" -m pip install pyyaml
exec "$ISAAC_SIM/python.sh" isaac/run_isaac.py "$@"
