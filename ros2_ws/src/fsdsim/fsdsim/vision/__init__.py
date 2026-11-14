"""Camera cone perception: YOLO cone detection on the ZED image, 3D cones from the depth image, a cone map,
automatic training labels from the simulator's ground truth, and accuracy against the ground truth.

Pure Python (NumPy, OpenCV, ONNX Runtime); the ROS 2 nodes are in fsd_sim_bridge (cone_perception,
cone_dataset_recorder). See docs/PERCEPTION.md.
"""

# YOLO class index == fsd_sim_msgs/Cone colour (BLUE=0, YELLOW=1, ORANGE_SMALL=2, ORANGE_BIG=3); FSOCO names
CLASS_NAMES = ["blue_cone", "yellow_cone", "orange_cone", "large_orange_cone"]
COLOR_NAMES = ["blue", "yellow", "orange", "big orange"]
