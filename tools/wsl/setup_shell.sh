#!/usr/bin/env bash
# One-time WSL shell setup for the FSG DV simulator, in the style of the Go2 setup:
#   * ROS 2 and the workspace are sourced only when you type `fsgsim` (fast new terminals)
#   * ROS 2 talks through the Zenoh router (rmw_zenoh_cpp), started with
#         fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd
#
#   bash ~/fsg-dv-sim/tools/wsl/setup_shell.sh            # install / update
#   bash ~/fsg-dv-sim/tools/wsl/setup_shell.sh --dry-run  # show what would change
#
# .bashrc is backed up first. Lines that this project's earlier instructions added
# (sourcing /opt/ros/jazzy and the fsg-dv-sim workspace, ROS_AUTOMATIC_DISCOVERY_RANGE)
# are commented out, not deleted. Other lines, e.g. your `go2sim` or `fs` aliases, are not touched.
set -euo pipefail

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
RC="$HOME/.bashrc"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BEGIN="# >>> fsg-dv-sim >>>"
END="# <<< fsg-dv-sim <<<"

block=$(cat <<EOF
$BEGIN
# FSG DV simulator: nothing is sourced here, only when you type 'fsgsim' (tools/wsl/fsgsim_env.sh)
export FSG_ROS_DOMAIN_ID=\${FSG_ROS_DOMAIN_ID:-18}
alias fsgsim='source "$REPO/tools/wsl/fsgsim_env.sh"'
alias fsgrouter='fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd'
alias fsgbuild='bash "$REPO/tools/wsl/fsgbuild.sh" && fsgsim'
$END
EOF
)

tmp=$(mktemp)
# 1) drop an older copy of our block, 2) comment out the global sources from earlier instructions
awk -v b="$BEGIN" -v e="$END" '
    # Lines inside a multi-line quoted alias (e.g. alias fs='...') belong to that alias:
    # leave them alone, and undo the commenting that version 0.4.0 of this script did there.
    inq {
        sub(/^# \[fsg-dv-sim: [^]]*\] /, "")
        print
        if (gsub(/\047/, "\047") % 2 == 1) inq = 0
        next
    }
    $0 == b { skip = 1; next }
    $0 == e { skip = 0; next }
    skip    { next }
    /^[[:space:]]*source \/opt\/ros\/jazzy\/setup\.bash[[:space:]]*$/                           { print "# [fsg-dv-sim: now in fsgsim] " $0; next }
    /fsg-dv-sim\/ros2_ws\/install\/setup\.bash/ && $0 !~ /^[[:space:]]*#/ && $0 !~ /\047/        { print "# [fsg-dv-sim: now in fsgsim] " $0; next }
    /^[[:space:]]*export ROS_AUTOMATIC_DISCOVERY_RANGE=[^\047]*$/                               { print "# [fsg-dv-sim: not used with Zenoh] " $0; next }
    {
        print
        if ($0 !~ /^[[:space:]]*#/ && gsub(/\047/, "\047") % 2 == 1) inq = 1
    }
' "$RC" > "$tmp"
# strip trailing blank lines so repeated runs do not grow the file, then append our block
body=$(cat "$tmp")
printf '%s\n\n%s\n' "$body" "$block" > "$tmp"

if [ "$DRY" = 1 ]; then
    diff -u "$RC" "$tmp" || true
    rm -f "$tmp"
    exit 0
fi

backup="$RC.bak.$(date +%Y%m%d-%H%M%S)"
cp "$RC" "$backup"
mv "$tmp" "$RC"
echo "Updated $RC (backup: $backup). Changes:"
diff -u "$backup" "$RC" || true

if [ ! -d /opt/ros/jazzy/share/rmw_zenoh_cpp ]; then
    echo
    echo "rmw_zenoh_cpp is not installed yet:  sudo apt install ros-jazzy-rmw-zenoh-cpp"
fi
cat <<'EOF'

Open a new terminal (or: source ~/.bashrc), then:
  terminal 1:  fsgrouter                                          # = fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd
  terminal 2:  fsgsim && ros2 launch fsd_sim_bringup isaac.launch.py
  Windows   :  run_isaac.ps1 -Mission skidpad
EOF
