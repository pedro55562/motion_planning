import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable


def bridge(robot_name):
    return ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            f'/{robot_name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            f'/{robot_name}/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            f'/{robot_name}/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            f'/{robot_name}/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            f'/{robot_name}/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
        ],
        output='screen',
    )


def spawn(robot_name, sdf_path, x, y):
    return ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-name', robot_name,
            '-file', sdf_path,
            '-x', str(x),
            '-y', str(y),
            '-z', '0.01',
        ],
        output='screen',
    )


def generate_launch_description():
    project_path = os.path.expanduser('~/projects/motion_planning/project_1')

    tb3_0_sdf = os.path.join(project_path, 'tb3_models', 'tb3_0', 'model.sdf')
    tb3_1_sdf = os.path.join(project_path, 'tb3_models', 'tb3_1', 'model.sdf')

    tb3_models_path = '/opt/ros/jazzy/share/turtlebot3_gazebo/models'

    return LaunchDescription([
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=[
                EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value=''),
                ':',
                tb3_models_path,
            ],
        ),

        ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'empty.sdf'],
            output='screen',
        ),

        TimerAction(
            period=2.0,
            actions=[
                spawn('tb3_0', tb3_0_sdf, 0.0, 0.0),
                spawn('tb3_1', tb3_1_sdf, 1.0, 0.0),
            ],
        ),

        TimerAction(
            period=4.0,
            actions=[
                bridge('tb3_0'),
                bridge('tb3_1'),
            ],
        ),
    ])