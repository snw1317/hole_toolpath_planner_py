from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    default_params = Path(__file__).resolve().parent.parent / "cfg" / "default_params.yaml"

    params_arg = DeclareLaunchArgument(
        "params",
        default_value=str(default_params),
        description="YAML file with python detector parameters",
    )

    detector_node = Node(
        package="hole_toolpath_planner_py",
        executable="hole_detector_py",
        name="hole_toolpath_planner_py",
        output="screen",
        parameters=[LaunchConfiguration("params")],
    )

    return LaunchDescription([params_arg, detector_node])
