from setuptools import setup

package_name = "hole_toolpath_planner_py"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/detector_py.launch.py"]),
        ("share/" + package_name + "/cfg", ["cfg/default_params.yaml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "open3d",
    ],
    zip_safe=True,
    maintainer="snw13",
    maintainer_email="snw13@example.com",
    description="Python-based hole detector using Open3D.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hole_detector_py = hole_toolpath_planner_py.detector_node:main",
        ],
    },
)
