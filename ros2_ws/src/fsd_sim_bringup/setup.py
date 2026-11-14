from glob import glob

from setuptools import find_packages, setup

package_name = "fsd_sim_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/tracks", glob("tracks/*.csv")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FS Driverless Team",
    maintainer_email="driverless@example.com",
    description="Launch files and configuration for the FSG driverless simulator",
    license="MIT",
)
