import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Launch configurations
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    container_name_full = (namespace, '/', container_name)

    # ... (declare launch arguments) ...
       # Declare launch arguments that can be overridden from the command line
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace', default_value='', description='Top-level namespace'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use simulation (Gazebo) clock'
    )

    # Rewritten parameters (example)
    param_substitutions = {'use_sim_time': use_sim_time}
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True)

    # ----- Non‑composed nodes -----
    load_nodes = GroupAction(
        condition=UnlessCondition(use_composition),
        actions=[
            Node(
                package='opennav_following',
                executable='following_server',
                name='following_server',
                namespace=namespace,
                output='screen',
                emulate_tty=True,
                parameters=[configured_params]   # your params file
            )
        ]
    )

    # ----- Composed nodes -----
    load_composable_nodes = GroupAction(
        condition=IfCondition(use_composition),
        actions=[
            LoadComposableNodes(
                target_container=container_name_full,
                composable_node_descriptions=[
                    ComposableNode(
                        package='opennav_following',
                        plugin='opennav_following::FollowingServer',
                        name='following_server',
                        parameters=[configured_params]   # your params file
                    ),
                ]
            )
        ]
    )


    ld = LaunchDescription()


    # Actions
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)



    return ld