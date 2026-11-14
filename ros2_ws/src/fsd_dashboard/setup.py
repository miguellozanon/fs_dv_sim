from glob import glob

from setuptools import find_packages, setup

package_name = "fsd_dashboard"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["web/*"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/web", glob("fsd_dashboard/web/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="FS Driverless Team",
    maintainer_email="driverless@example.com",
    description="Live web dashboard for the FSG driverless simulator / digital twin",
    license="MIT",
    entry_points={
        "console_scripts": [
            "dashboard = fsd_dashboard.node:main",
            "dashboard_demo = fsd_dashboard.demo:main",
        ],
    },
)
