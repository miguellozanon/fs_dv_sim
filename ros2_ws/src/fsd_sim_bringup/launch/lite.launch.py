"""Lightweight backend (no GPU): simulator + static TFs + reference driver + RViz.

    ros2 launch fsd_sim_bringup lite.launch.py mission:=skidpad
    ros2 launch fsd_sim_bringup lite.launch.py mission:=trackdrive driver:=false   # your stack drives
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    from fsd_sim_bridge.common import zenoh_router_problem  # noqa: PLC0415

    problem = zenoh_router_problem()
    if problem:
        return [LogInfo(msg="[fsd] " + problem), Shutdown(reason="no Zenoh router")]
    share = get_package_share_directory("fsd_sim_bringup")
    cfg_dir = LaunchConfiguration("config_dir").perform(context) or os.path.join(share, "config")
    mission = LaunchConfiguration("mission").perform(context)
    track_file = LaunchConfiguration("track_file").perform(context)
    seed = int(LaunchConfiguration("seed").perform(context))
    rtf = float(LaunchConfiguration("realtime_factor").perform(context))
    use_rviz = LaunchConfiguration("rviz").perform(context).lower() == "true"
    rviz_sim_time = "true" if LaunchConfiguration("rviz_sim_time").perform(context).lower() == "true" else "false"
    use_driver = LaunchConfiguration("driver").perform(context).lower() == "true"
    driver_mode = LaunchConfiguration("driver_mode").perform(context)

    actions = [
        Node(
            package="fsd_sim_bridge",
            executable="lite_sim",
            name="fsd_lite_sim",
            output="screen",
            parameters=[{
                "config_dir": cfg_dir,
                "mission": mission,
                "track_file": track_file,
                "seed": seed,
                "realtime_factor": rtf,
                "vehicle_set": LaunchConfiguration("vehicle_set").perform(context),
                "faults": LaunchConfiguration("faults").perform(context),
            }],
        )
    ]
    perception = LaunchConfiguration("perception").perform(context).lower()
    p_nodes, drv_params = _perception_nodes(context, perception, cfg_dir, isaac=False)
    actions += p_nodes
    if use_driver:
        actions.append(
            Node(
                package="fsd_sim_bridge",
                executable="demo_driver",
                name="fsd_demo_driver",
                output="screen",
                parameters=[{"use_sim_time": True, "mode": driver_mode,
                             "longitudinal": LaunchConfiguration("longitudinal").perform(context),
                             "rl_policy": LaunchConfiguration("rl_policy").perform(context), **drv_params}],
            )
        )
    if use_rviz:
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                # Started exactly like
                #   ros2 run rviz2 rviz2 -d fsd_sim.rviz --ros-args -p use_sim_time:=false
                # No node name override: it adds a __node remap, which renames every node inside the
                # RViz process. Wall clock by default: RViz draws sim-stamped data fine and cannot
                # lose its TF buffer to /clock jumps. rviz_sim_time:=true restores the textbook setup.
                arguments=["-d", os.path.join(share, "rviz", "fsd_sim.rviz")],
                ros_arguments=["-p", "use_sim_time:=" + rviz_sim_time],
                output="screen",
            )
        )
    return actions


def _perception_nodes(context, perception, cfg_dir, isaac):
    """Nodes and driver parameters for perception:=lidar | fusion | camera (fsd_perception) and yolo."""
    slam = LaunchConfiguration("slam").perform(context).lower() == "true"
    nodes, drv = [], {"line": LaunchConfiguration("line").perform(context),
                      "race_after": int(LaunchConfiguration("race_after").perform(context) or 0)}
    if perception in ("lidar", "fusion", "camera"):
        colour = "/perception/frame_cones" if isaac else "/zed/detections"
        if isaac and perception in ("fusion", "camera"):
            # colours (and camera cones) from YOLO on the rendered ZED image
            nodes.append(Node(package="fsd_sim_bridge", executable="cone_perception", name="fsd_cone_perception", output="screen",
                              parameters=[{"use_sim_time": True, "model": LaunchConfiguration("cone_model").perform(context),
                                           "cones_topic": "/perception/camera_cones"}]))
        params = {"use_sim_time": True, "mode": perception, "slam": slam, "colour_topic": colour}
        if cfg_dir:
            params["config_dir"] = cfg_dir
        nodes.append(Node(package="fsd_sim_bridge", executable="perception", name="fsd_perception", output="screen",
                          parameters=[params]))
        drv.update(cones_topic="/perception/cones", odom_topic="/slam/odom" if slam else "/ground_truth/odom")
    return nodes, drv


def _dashboard(context):
    if LaunchConfiguration("dashboard").perform(context).lower() != "true":
        return []
    return [Node(
        package="fsd_dashboard",
        executable="dashboard",
        name="fsd_dashboard",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "port": int(LaunchConfiguration("dashboard_port").perform(context)),
            "source": "Lite simulator",
            "start_setup": LaunchConfiguration("vehicle_setup").perform(context),
        }],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mission", default_value="", description="acceleration | skidpad | autocross | trackdrive | ebs_test (empty: sim.yaml)"),
        DeclareLaunchArgument("config_dir", default_value="", description="directory with vehicle/sensors/sim.yaml"),
        DeclareLaunchArgument("track_file", default_value="", description="optional EUFS-style track CSV"),
        DeclareLaunchArgument("seed", default_value="-1", description="track generator seed (-1: sim.yaml)"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0"),
        DeclareLaunchArgument("vehicle_set", default_value="",
                              description='vehicle parameter changes, e.g. "m=300 mux*1.05" (LapSim names, see vehicle.yaml)'),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_sim_time", default_value="false", description="run RViz on /clock"),
        DeclareLaunchArgument("driver", default_value="true", description="start the reference driver"),
        DeclareLaunchArgument("driver_mode", default_value="auto", description="auto | map | cones"),
        DeclareLaunchArgument("longitudinal", default_value="reference",
                              description="throttle / brake: reference | rl (learned policy, tools/train_rl.py)"),
        DeclareLaunchArgument("rl_policy", default_value="auto",
                              description="auto = the policy trained for the running vehicle spec, or a name in rl/policies"),
        DeclareLaunchArgument("perception", default_value="none",
                              description="none: ground-truth cones; lidar: LiDAR cones (colours from the intensity stripes); "
                                          "fusion: LiDAR cones + camera colours; camera: camera cones only"),
        DeclareLaunchArgument("line", default_value="center",
                              description="map mode on circuits: center | racing (racing line inside the cones, fsdsim.perf.raceline)"),
        DeclareLaunchArgument("slam", default_value="false",
                              description="with perception lidar / fusion / camera: the driver's pose also comes from the perception "
                                          "node (wheel speeds + gyro + cone map) instead of the ground truth"),
        DeclareLaunchArgument("race_after", default_value="0",
                              description="circuits in cones mode: after this many laps drive the mapped track (with line:=racing its racing line)"),
        DeclareLaunchArgument("faults", default_value="",
                              description='sensor fault injection, e.g. "lidar:dropout@20+3,gnss:jump=1.5@10+30" (docs/PERCEPTION.md)'),
        DeclareLaunchArgument("dashboard", default_value="true", description="web dashboard on http://localhost:<dashboard_port>"),
        DeclareLaunchArgument("dashboard_port", default_value="8080"),
        DeclareLaunchArgument("vehicle_setup", default_value="",
                              description="the DV launcher's vehicle setup for the dashboard (base64 JSON)"),
        OpaqueFunction(function=_setup),
        OpaqueFunction(function=_dashboard),
    ])
