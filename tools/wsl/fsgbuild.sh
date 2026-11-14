#!/usr/bin/env bash
# Build the ROS 2 workspace of this repo (the `fsgbuild` alias; the launcher's "Build workspace").
#
# For a repo on a Windows drive (/mnt/c/formula_student/fsg-dv-sim) colcon's build/ install/ log/
# go to ~/.fsg-dv-sim/colcon on the Linux disk: fast, and symlink-install works there. The sources
# stay on C:, so edits in Windows are live (Python, launch and config files need no rebuild; new
# files and setup.py / package.xml changes do - the launcher rebuilds then by itself).
#   FSG_WS / FSG_COLCON_BASE: see fsgsim_env.sh
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${FSG_WS:-$REPO/ros2_ws}"
if [ -n "${FSG_COLCON_BASE:-}" ]; then CB="$FSG_COLCON_BASE"
elif [[ "$WS" == /mnt/* ]]; then CB="$HOME/.fsg-dv-sim/colcon"
else CB="$WS"; fi
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    echo "fsgbuild: ROS 2 Jazzy is not installed (/opt/ros/jazzy)" >&2
    exit 1
fi
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
mkdir -p "$CB"
cd "$CB"
echo "[fsgbuild] sources $WS/src -> build / install / log in $CB"
colcon build --symlink-install --base-paths "$WS/src" --build-base build --install-base install "$@"
python3 "$REPO/tools/launcher/launcher.py" --write-build-stamp >/dev/null 2>&1 || true
echo "[fsgbuild] done"
