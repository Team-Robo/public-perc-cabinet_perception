import os
from glob import glob

from setuptools import find_packages, setup

package_name = "cabinet_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Cabinet boards, cubbies, shelf members and free space -> CabinetScene.",
    maintainer="Nanyang Technological University Team Robo",
    maintainer_email="teamrobontu@gmail.com",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cabinet_node = cabinet_perception.cabinet_node:main",
            "synthetic_camera = cabinet_perception.synthetic_camera:main",
            "cabinet_demo = cabinet_perception.demo:main",
            "handheld_tf = cabinet_perception.handheld_tf:main",
        ],
    },
)
