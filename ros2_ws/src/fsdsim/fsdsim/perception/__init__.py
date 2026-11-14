"""Perception without ground truth: cones from the LiDAR, colours from the camera, cone map and pose (SLAM).

    lidar_cones.LidarConeDetector   point cloud -> cones in the car frame (own-car filter, ground plane, clusters)
    fusion.fuse                     LiDAR positions + camera colours
    slam.ConeSlam                   wheel speed + gyro odometry, corrected on the cone map it builds
    pipeline.PerceptionPipeline     the three together, fed with simulator events (used by the tools and ROS nodes)
"""

from .fusion import camera_to_base, fuse  # noqa: F401
from .lidar_cones import LidarConeDetector  # noqa: F401
from .pipeline import PerceptionPipeline  # noqa: F401
from .slam import ConeSlam  # noqa: F401

BLUE, YELLOW, ORANGE_SMALL, ORANGE_BIG, UNKNOWN = 0, 1, 2, 3, 4
