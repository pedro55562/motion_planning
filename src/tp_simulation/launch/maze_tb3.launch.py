#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    x_pose = LaunchConfiguration("x_pose", default="0.250")
    y_pose = LaunchConfiguration("y_pose", default="21.650")

    tp_simulation_dir = get_package_share_directory("tp_simulation")
    turtlebot3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    ros_gz_sim_dir = get_package_share_directory("ros_gz_sim")

    launch_file_dir = os.path.join(turtlebot3_gazebo_dir, "launch")

    world = os.path.join(
        tp_simulation_dir,
        "worlds",
        "maze.world",
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-r -s -v2 ", world],
            "on_exit_shutdown": "true",
        }.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": "-g -v2 ",
        }.items(),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "robot_state_publisher.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={
            "x_pose": x_pose,
            "y_pose": y_pose,
        }.items(),
    )

    # Gazebo needs both:
    #   1. your custom map model: map_maze
    #   2. TurtleBot3 model files
    add_tp_models_path = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(tp_simulation_dir, "models"),
    )

    add_tb3_models_path = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(turtlebot3_gazebo_dir, "models"),
    )

    ld = LaunchDescription()

    ld.add_action(add_tp_models_path)
    ld.add_action(add_tb3_models_path)

    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(robot_state_publisher_cmd)

    return ld
