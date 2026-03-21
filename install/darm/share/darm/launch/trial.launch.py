import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import xacro

def generate_launch_description():

    package_name = 'darm'
    pkg_project_description = get_package_share_directory(package_name)

    xacro_file = os.path.join(pkg_project_description, 'urdf', 'x500_base', 'robot.urdf.xacro')
    doc = xacro.process_file(xacro_file)

    robot_desc = doc.toxml().replace(
        "package://darm",
        pkg_project_description
    )

    params = {'robot_description': robot_desc, 'use_sim_time': True}
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare('darm'),
            'config',
            'arm_control.yaml',
        ]
    )
    
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # rviz = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     output='screen',
    #     parameters=[{'use_sim_time': True}],
    # )

    use_gpu = os.environ.get('USE_GPU', '1') == '1'
    launch_env_vars = []
    if use_gpu:
        launch_env_vars.extend([
            SetEnvironmentVariable(name='__NV_PRIME_RENDER_OFFLOAD', value='1'),
            SetEnvironmentVariable(name='__GLX_VENDOR_LIBRARY_NAME', value='nvidia'),
        ])

    # gzbridge = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     arguments=[
    #         '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
    #         "/world/default/model/x500_0/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model"
    #     ],
    #     remappings=[
    #         ('/world/default/model/x500_0/joint_state', '/joint_states'),
    #     ],
    # )

    # Wait longer for PX4 to fully initialize (20 seconds)
    # Spawn joint_state_broadcaster FIRST and ALONE
    spawn_joint_state_broadcaster = TimerAction(
        period=20.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'joint_state_broadcaster',
                    '--controller-manager-timeout', '30'  # Longer timeout
                ],  
                output='screen'
            )
        ]
    )

    spawn_arm = TimerAction(
        period=30.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'arm_effort_controller',
                    '--param-file', robot_controllers,
                    '--controller-manager-timeout', '30'
                ],
                output='screen'
            )
        ]
    )
    # spawn_arm = TimerAction(
    #     period=30.0,
    #     actions=[
    #         Node(
    #             package='controller_manager',
    #             executable='spawner',
    #             arguments=[
    #                 'arm_position_controller',
    #                 '--param-file', robot_controllers,
    #                 '--controller-manager-timeout', '30'
    #             ],
    #             output='screen'
    #         )
    #     ]
    # )

    # spawn_shoulder = TimerAction(
    #     period=30.0,
    #     actions=[
    #         Node(
    #             package='controller_manager',
    #             executable='spawner',
    #             arguments=[
    #                 'shoulder_effort_controller',
    #                 '--param-file', robot_controllers,
    #                 '--controller-manager-timeout', '30'
    #             ],
    #             output='screen'
    #         )
    #     ]
    # )

    # spawn_elbow = TimerAction(
    #     period=35.0,
    #     actions=[
    #         Node(
    #             package='controller_manager',
    #             executable='spawner',
    #             arguments=[
    #                 'elbow_effort_controller',
    #                 '--param-file', robot_controllers,
    #                 '--controller-manager-timeout', '30'
    #             ],
    #             output='screen'
    #         )
    #     ]
    # )

    # spawn_jaw = TimerAction(
    #     period=40.0,
    #     actions=[
    #         Node(
    #             package='controller_manager',
    #             executable='spawner',
    #             arguments=[
    #                 'jaw_effort_controller',
    #                 '--param-file', robot_controllers,
    #                 '--controller-manager-timeout', '30'
    #             ],
    #             output='screen'
    #         )
    #     ]
    # )

    return LaunchDescription([
        *launch_env_vars,
        rsp,
        # rviz,
        # gzbridge,         
        spawn_joint_state_broadcaster,
        spawn_arm,
        # spawn_shoulder,
        # spawn_elbow,
        # spawn_jaw
    ])
