"""ROS 2 side of the Isaac Sim backend (run in WSL / Linux): relay + reference driver + RViz.

The relay listens on TCP port 47800; Isaac Sim (Windows or Linux) connects to it
and sends its full configuration, so nothing has to be kept in sync here.

    ros2 launch fsd_sim_bringup isaac.launch.py
    ros2 launch fsd_sim_bringup isaac.launch.py driver:=false port:=47801

Then on Windows (PowerShell):
    .\\isaac\\run_isaac.ps1 -Mission skidpad

RViz is started once the relay has verified that TF arrives over ROS 2 (after Isaac
connects), which is exactly the situation in which a hand-started RViz works.
rviz_wait:=false starts it immediately instead.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessIO
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

READY_MARK = "TF self-check"  # logged by the relay ~4 s after Isaac connects (OK or FAILED)


def _flag(context, name: str) -> bool:
    return LaunchConfiguration(name).perform(context).lower() == "true"


def _setup(context):
    from fsd_sim_bridge.common import zenoh_router_problem  # noqa: PLC0415

    problem = zenoh_router_problem()
    if problem:
        return [LogInfo(msg="[fsd] " + problem), Shutdown(reason="no Zenoh router")]
    share = get_package_share_directory("fsd_sim_bringup")
    rviz_sim_time = "true" if _flag(context, "rviz_sim_time") else "false"
    driver_mode = LaunchConfiguration("driver_mode").perform(context)

    relay = Node(
        package="fsd_sim_bridge",
        executable="isaac_relay",
        name="fsd_isaac_relay",
        output="screen",
        parameters=[{
            # The relay is the /clock source, so it must run on wall time itself.
            "use_sim_time": False,
            "bind_host": LaunchConfiguration("bind_host").perform(context),
            "port": int(LaunchConfiguration("port").perform(context)),
            "publish_clock": LaunchConfiguration("publish_clock").perform(context),
            "expect_rviz": _flag(context, "rviz"),
        }],
    )
    actions = [relay]
    perception = LaunchConfiguration("perception").perform(context).lower()
    p_nodes, drv_params = _perception_nodes(context, perception, "", isaac=True)
    actions += p_nodes
    if perception == "yolo":
        # camera cone detection (YOLO + depth) -> /perception/cones; the driver uses it instead of the ground truth
        actions.append(Node(
            package="fsd_sim_bridge", executable="cone_perception", name="fsd_cone_perception", output="screen",
            parameters=[{"use_sim_time": True, "model": LaunchConfiguration("cone_model").perform(context)}]))
    if _flag(context, "record_cones"):
        actions.append(Node(
            package="fsd_sim_bridge", executable="cone_dataset_recorder", name="fsd_cone_dataset_recorder",
            output="screen", parameters=[{"use_sim_time": True,
                                          "session": LaunchConfiguration("record_session").perform(context)}]))
    if _flag(context, "driver"):
        actions.append(
            Node(
                package="fsd_sim_bridge",
                executable="demo_driver",
                name="fsd_demo_driver",
                output="screen",
                parameters=[{"use_sim_time": True, "mode": driver_mode,
                             "longitudinal": LaunchConfiguration("longitudinal").perform(context),
                             "rl_policy": LaunchConfiguration("rl_policy").perform(context),
                             "cones_topic": "/perception/cones" if perception == "yolo" else "/ground_truth/cones",
                             "latency": float(LaunchConfiguration("driver_latency").perform(context)),
                             "tick_clock": "steady",
                             **drv_params}],
            )
        )
    if not _flag(context, "rviz"):
        return actions

    def make_rviz():
        return Node(
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

    if not _flag(context, "rviz_wait"):
        actions.append(make_rviz())
        return actions

    started = {"rviz": False}

    def on_relay_output(event):
        if started["rviz"] or READY_MARK not in event.text.decode(errors="ignore"):
            return None
        started["rviz"] = True
        return [LogInfo(msg="[fsd] Isaac Sim data is flowing - starting RViz"), make_rviz()]

    actions += [
        LogInfo(msg="[fsd] RViz starts once Isaac Sim is connected and TF is verified "
                    "(start isaac/run_isaac.ps1 on Windows; rviz_wait:=false starts it right away)"),
        # ROS loggers write to stderr; watch both streams.
        RegisterEventHandler(OnProcessIO(target_action=relay, on_stdout=on_relay_output,
                                         on_stderr=on_relay_output)),
    ]
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
        if isaac:
            params["lidar_time_offset"] = float(LaunchConfiguration("lidar_time_offset").perform(context))
            # the RTX LiDAR has no stripe intensities: colours guessed from them were right for 23 % of the cones
            # (11 Nov 2026); without colours the driver follows the boundaries by geometry
            params["lidar_colours"] = False
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
            "source": "Isaac Sim via relay",
            "start_setup": LaunchConfiguration("vehicle_setup").perform(context),
            "camera_hz": float(LaunchConfiguration("camera_hz").perform(context) or 0.0),
        }],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("bind_host", default_value="0.0.0.0", description="interface the relay listens on"),
        DeclareLaunchArgument("port", default_value="47800", description="TCP port Isaac Sim connects to"),
        DeclareLaunchArgument("publish_clock", default_value="auto",
                              description="auto: relay publishes /clock unless Isaac uses the ros2 sensor transport"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_wait", default_value="true",
                              description="start RViz only after Isaac connected and TF was verified"),
        DeclareLaunchArgument("rviz_sim_time", default_value="false", description="run RViz on /clock"),
        DeclareLaunchArgument("driver", default_value="true"),
        DeclareLaunchArgument("driver_mode", default_value="auto"),
        DeclareLaunchArgument("longitudinal", default_value="reference",
                              description="throttle / brake: reference | rl (learned policy, tools/train_rl.py)"),
        DeclareLaunchArgument("rl_policy", default_value="auto",
                              description="auto = the policy trained for the running vehicle spec, or a name in rl/policies"),
        DeclareLaunchArgument("perception", default_value="none",
                              description="none: ground-truth cones; yolo: camera cones (YOLO, ground-truth pose); fusion: LiDAR "
                                          "cones + YOLO colours; lidar: LiDAR cones only (no colours from the RTX LiDAR: use "
                                          "fusion); camera: YOLO cones through the SLAM node. Faults for Isaac's own sensors: "
                                          "run_isaac.ps1 -Faults"),
        DeclareLaunchArgument("line", default_value="center",
                              description="map mode on circuits: center | racing (racing line inside the cones, fsdsim.perf.raceline)"),
        DeclareLaunchArgument("slam", default_value="false",
                              description="with perception lidar / fusion / camera: the driver's pose also comes from the perception "
                                          "node (wheel speeds + gyro + cone map) instead of the ground truth"),
        DeclareLaunchArgument("race_after", default_value="0",
                              description="circuits in cones mode: after this many laps drive the mapped track (with line:=racing its racing line)"),

        DeclareLaunchArgument("driver_latency", default_value="0.07",
                              description="s the reference driver predicts its pose ahead: Isaac renders at 30 Hz and a command "
                                          "acts one frame later (without it the skidpad failed from lap 3, 11 Nov 2026)"),
        DeclareLaunchArgument("lidar_time_offset", default_value="0.0",
                              description="s added to the RTX LiDAR scan stamps in the perception node; the dashboard's map line "
                                          "shows the best value measured against the ground truth (lidar timing)"),
        DeclareLaunchArgument("cone_model", default_value="", description="YOLO model folder (default: newest in models/cones)"),
        DeclareLaunchArgument("record_cones", default_value="false",
                              description="record camera images + cone labels for YOLO training (datasets/cones)"),
        DeclareLaunchArgument("record_session", default_value="", description="dataset session name (default: date)"),
        DeclareLaunchArgument("dashboard", default_value="true", description="web dashboard on http://localhost:<dashboard_port>"),
        DeclareLaunchArgument("dashboard_port", default_value="8080"),
        DeclareLaunchArgument("vehicle_setup", default_value="",
                              description="the DV launcher's vehicle setup for the dashboard (base64 JSON)"),
        DeclareLaunchArgument("camera_hz", default_value="0",
                              description="camera rate Isaac was started with (run_isaac.py --camera-hz); 0: sensors.yaml. "
                                          "The dashboard's stream health expects this rate"),
        OpaqueFunction(function=_setup),
        OpaqueFunction(function=_dashboard),
    ])
