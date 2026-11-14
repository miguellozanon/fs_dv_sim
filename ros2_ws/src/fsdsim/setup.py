from setuptools import find_packages, setup

package_name = "fsdsim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="FS Driverless Team",
    maintainer_email="driverless@example.com",
    description="Simulator core for the FSG driverless digital twin",
    license="MIT",
)
