import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import LoadComposableNodes, Node, PushROSNamespace, SetParameter
from launch_ros.descriptions import ComposableNode



def generate_launch_description() -> LaunchDescription:
    # Get the launch directory
    navigation_pkg_path = get_package_share_directory('mobo_bot_navigation')

    # Launch configurations (set via CLI or default values)
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    #docking_yaml_file = LaunchConfiguration('docking_yaml_file')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    #use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')

    # Fixed values (not user-configurable)
    autostart = True

    # Build the path to the keepout mask YAML file (static inside your package)
    docking_path = PathJoinSubstitution([
        navigation_pkg_path,
        "config",
        PythonExpression(expression=["'", "docking", "'", " + '.yaml'"])
    ])



    # Remappings for tf topics
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    # # Value rewrites for the main params file (e.g., to enable/disable keepout zones)
    # yaml_substitutions = {
    #     'x': init_pose_x,
    #     'y': init_pose_y,
    #     'yaw': init_pose_yaw
    # }
   
    # # Load the main kaapout zones parameters file, applying value rewrites and namespace prefix
    # configured_params = RewrittenYaml(
    #     source_file=params_file,
    #     root_key=namespace,
    #     param_rewrites=yaml_substitutions,
    #     convert_types=True)

    # Environment variable to fix log buffering
    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1'
    )

    # Declare launch arguments that can be overridden from the command line
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace', default_value='', description='Top-level namespace'
    )

    # declare_keepout_mask_yaml_cmd = DeclareLaunchArgument(
    #     'docking_yaml_file',
    #     default_value= docking_path,
    #     description='Full path to speed mask yaml file to load',
    # )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='True', description='Use simulation (Gazebo) clock'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(navigation_pkg_path, 'config', 'docking.yaml'),
        description='Full path to the ROS2 parameters file for all launched nodes',
    )

    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition', default_value='False', description='Use composed bringup if True'
    )

    declare_container_name_cmd = DeclareLaunchArgument(
        'container_name', default_value='nav2_container', description='Name of the composable container'
    )

    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn', default_value='False', description='Respawn nodes if they crash (non‑composition only)'
    )

    # declare_use_keepout_zones_cmd = DeclareLaunchArgument(
    #     'use_keepout_zones', default_value='True', description='Enable or disable keepout zones'
    # )

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', default_value='info', description='Log level for nodes'
    )

    # ----- Non‑composed nodes (standalone processes) -----
    load_nodes = GroupAction(
        condition=IfCondition(PythonExpression(['not ', use_composition])),
        actions=[
            PushROSNamespace(namespace),
            SetParameter('use_sim_time', use_sim_time),
            # Map server that publishes the keepout mask
             Node(
                package='opennav_docking',
                executable='opennav_docking',
                name='docking_server',
                output='screen',
                parameters=[docking_path],
            ),
        
            # Lifecycle manager for the eocking server
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_docking',
                output='screen',
                arguments=['--ros-args', '--log-level', log_level],
                parameters=[{'autostart': autostart, 'use_sim_time': use_sim_time,}, {'node_names': ['docking_server']}],
            ),
        ],
    )

    # ----- Composed nodes (inside a container) -----
    container_name_full = (namespace, '/', container_name)

    load_composable_nodes = GroupAction(
        condition=IfCondition(use_composition),   # start only if composition is ON
        actions=[
            PushROSNamespace(namespace),
            SetParameter('use_sim_time', use_sim_time),
            LoadComposableNodes(
                target_container=container_name_full,
                composable_node_descriptions=[
                    ComposableNode(
                        package='opennav_docking',
                        plugin='opennav_docking::DockingServer',
                        name='docking_server',
                        #output='screen',
                        parameters=[params_file],
                    ),
                ],
            ),
            # Load the lifecycle manager as a composable node
            LoadComposableNodes(
                target_container=container_name_full,
                composable_node_descriptions=[
                    ComposableNode(
                        package='nav2_lifecycle_manager',
                        plugin='nav2_lifecycle_manager::LifecycleManager',
                        name='lifecycle_manager_docking',
                        #output='screen',
                        parameters=[{'autostart': autostart, 'use_sim_time': use_sim_time,}, {'node_names': ['docking_server']}],
                    ),
                ],
            ),
        ],
    )




    # load_composable_nodes = GroupAction(
    #     condition=IfCondition(use_composition),   # start only if composition is ON
    #    actions=[
    #         PushROSNamespace(namespace),
    #         SetParameter('use_sim_time', use_sim_time),
    #         # Map server that publishes the keepout mask
    #          Node(
    #             package='opennav_docking',
    #             executable='opennav_docking',
    #             name='docking_server',
    #             output='screen',
    #             parameters=[params_file],
    #         ),
        
    #         # Lifecycle manager for the two keepout components
    #         Node(
    #             package='nav2_lifecycle_manager',
    #             executable='lifecycle_manager',
    #             name='lifecycle_manager_docking',
    #             output='screen',
    #             arguments=['--ros-args', '--log-level', log_level],
    #             parameters=[{'autostart': autostart, 'use_sim_time': use_sim_time,}, {'node_names': ['docking_server']}],
    #         ),
    #     ],
    # )

    # ----- Build the launch description -----
    ld = LaunchDescription()

    # Environment
    ld.add_action(stdout_linebuf_envvar)

    # Declared arguments
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_container_name_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_log_level_cmd)

    # Actions
    ld.add_action(load_nodes)
    ld.add_action(load_composable_nodes)

    return ld

