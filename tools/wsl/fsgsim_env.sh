# FSG DV simulator environment for WSL - sourced by the `fsgsim` alias, never from .bashrc
# directly, so new terminals stay fast (same idea as `go2sim` in the Go2 setup).
#
#   fsgsim                                         # this environment in the current terminal
#   fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd    # Zenoh router (one per machine; also serves Go2)
#   fsgsim && ros2 launch fsd_sim_bringup isaac.launch.py
#
# Settings (export them before calling fsgsim, or edit ~/.bashrc):
#   FSG_ROS_DOMAIN_ID   ROS domain of the simulator (default 18; Go2 uses 17, so both can run at once)
#   FSG_WS              workspace sources (default: ros2_ws of the repo this file is in)
#   FSG_COLCON_BASE     where colcon's build/ install/ log/ go (default: the workspace itself; for a
#                       repo on a Windows drive (/mnt/c/...) ~/.fsg-dv-sim/colcon, on the fast Linux disk)
#   FSG_RMW             rmw_zenoh_cpp (default) or rmw_fastrtps_cpp

if [ -z "${BASH_VERSION:-}" ]; then
    echo "fsgsim: source this from bash" >&2
    return 1 2>/dev/null || exit 1
fi

_fsg_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_fsg_ws="${FSG_WS:-$_fsg_repo/ros2_ws}"
if [ -n "${FSG_COLCON_BASE:-}" ]; then
    _fsg_cb="$FSG_COLCON_BASE"
elif [[ "$_fsg_ws" == /mnt/* ]]; then
    _fsg_cb="$HOME/.fsg-dv-sim/colcon"
else
    _fsg_cb="$_fsg_ws"
fi
export FSG_REPO="$_fsg_repo"

if [ -f /opt/ros/jazzy/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
else
    echo "fsgsim: /opt/ros/jazzy/setup.bash not found (is ROS 2 Jazzy installed?)" >&2
    return 1
fi

export RMW_IMPLEMENTATION="${FSG_RMW:-rmw_zenoh_cpp}"
export ROS_DOMAIN_ID="${FSG_ROS_DOMAIN_ID:-18}"
# Discovery goes through the Zenoh router, not DDS multicast: these DDS-only knobs do not apply.
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_LOCALHOST_ONLY

if [ -f "$_fsg_cb/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$_fsg_cb/install/setup.bash"
    _fsg_ws_state="workspace sourced ($_fsg_repo)"
else
    _fsg_ws_state="workspace NOT built: no $_fsg_cb/install/setup.bash - run fsgbuild"
fi

if [ "$RMW_IMPLEMENTATION" = "rmw_zenoh_cpp" ]; then
    if [ ! -d /opt/ros/jazzy/share/rmw_zenoh_cpp ]; then
        echo "fsgsim: rmw_zenoh_cpp is not installed: sudo apt install ros-jazzy-rmw-zenoh-cpp" >&2
    # Look for a listening socket instead of connecting: with WSL "mirrored" networking a test
    # connection to 127.0.0.1 can hang or be answered by Windows, which made this check unreliable.
    elif command -v ss >/dev/null 2>&1; then
        if [ -n "$(ss -Hltn 'sport = :7447' 2>/dev/null)" ]; then
            _fsg_router="Zenoh router on 127.0.0.1:7447"
        else
            _fsg_router="no Zenoh router yet - start it in its own terminal: fsgrouter"
        fi
    elif timeout 1 bash -c 'exec 3<>/dev/tcp/127.0.0.1/7447' 2>/dev/null; then
        _fsg_router="Zenoh router on 127.0.0.1:7447"
    else
        _fsg_router="no Zenoh router yet - start it in its own terminal: fsgrouter"
    fi
fi

echo "[fsgsim] ${WSL_DISTRO_NAME:+$WSL_DISTRO_NAME | }ROS 2 $ROS_DISTRO | RMW $RMW_IMPLEMENTATION | domain $ROS_DOMAIN_ID | $_fsg_ws_state${_fsg_router:+ | $_fsg_router}"
unset _fsg_repo _fsg_ws _fsg_cb _fsg_ws_state _fsg_router
return 0
