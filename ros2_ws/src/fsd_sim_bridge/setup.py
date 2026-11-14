from setuptools import find_packages, setup

package_name = "fsd_sim_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FS Driverless Team",
    maintainer_email="driverless@example.com",
    description="ROS 2 nodes for the FSG driverless simulator",
    license="MIT",
    entry_points={
        "console_scripts": [
            "lite_sim = fsd_sim_bridge.lite_sim_node:main",
            "isaac_relay = fsd_sim_bridge.isaac_relay_node:main",
            "demo_driver = fsd_sim_bridge.demo_driver_node:main",
            "cone_perception = fsd_sim_bridge.cone_perception_node:main",
            "perception = fsd_sim_bridge.perception_node:main",
            "cone_dataset_recorder = fsd_sim_bridge.cone_dataset_recorder_node:main",
        ],
    },
)
