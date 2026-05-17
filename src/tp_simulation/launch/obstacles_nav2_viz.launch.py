#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    yaw_pose = LaunchConfiguration("yaw_pose")

    tp_simulation_dir = get_package_share_directory("tp_simulation")

    map_yaml = os.path.join(
        tp_simulation_dir,
        "maps",
        "obstacles.yaml",
    )

    rviz_config = os.path.join(
        tp_simulation_dir,
        "rviz",
        "maze_nav2.rviz",
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"yaml_filename": map_yaml},
        ],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )

    static_map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_map_to_odom",
        arguments=[
            "--x", x_pose,
            "--y", y_pose,
            "--z", "0.0",
            "--yaw", yaw_pose,
            "--pitch", "0.0",
            "--roll", "0.0",
            "--frame-id", "map",
            "--child-frame-id", "odom",
        ],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("x_pose", default_value="7.525"),
            DeclareLaunchArgument("y_pose", default_value="3.475"),
            DeclareLaunchArgument("yaw_pose", default_value="0.0"),
            map_server,
            lifecycle_manager,
            static_map_to_odom,
            rviz,
        ]
    )
