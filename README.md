# hole_toolpath_planner_py

Python-only hole detection pipeline built on Open3D for the Hole Toolpath Planner workspace.

## Prerequisites
- Workspace already built at least once with `colcon build`
- ROS 2 Humble environment sourced before building or running

## Build
```bash
cd ~/hole_toolpath_planner_ws
colcon build --packages-select hole_toolpath_planner_py
```

## Run The Detector Node
Use two terminals so the detector keeps spinning while you issue service calls.

**Terminal 1**
```bash
cd ~/hole_toolpath_planner_ws
source install/setup.bash
ros2 launch hole_toolpath_planner_py detector_py.launch.py
```

**Terminal 2**
```bash
cd ~/hole_toolpath_planner_ws
source install/setup.bash
ros2 service call /detect_holes hole_toolpath_planner/srv/DetectHoles "{mesh_path: '/home/snw13/hole_toolpath_planner_ws/src/hole_toolpath_planner/test_parts/hole_test_plate_m.stl', min_diameter: 0.004, max_diameter: 0.006, min_length: 0.002, watertight_hint: true}"
```

The call above should return eleven 5 mm holes whose ground-truth values are listed in `hole_toolpath_planner/test_parts/hole_results.csv`.
