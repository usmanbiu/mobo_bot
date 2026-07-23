import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import LoadComposableNodes, Node, PushROSNamespace, SetParameter
from launch_ros.descriptions import ComposableNode
from nav2_common.launch import RewrittenYaml


def generate_launch_description() -> LaunchDescription:
    # Get the launch directory
    navigation_pkg_path = get_package_share_directory('mobo_bot_navigation')

    # Launch configurations (set via CLI or default values)
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    keepout_mask_yaml_file = LaunchConfiguration('keepout_mask_yaml_file')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    use_respawn = LaunchConfiguration('use_respawn')
    use_keepout_zones = LaunchConfiguration('use_keepout_zones')
    log_level = LaunchConfiguration('log_level')

    # Fixed values (not user-configurable)
    autostart = True

    # Build the path to the keepout mask YAML file (static inside your package)
    keepout_mask_path = PathJoinSubstitution([
        navigation_pkg_path,
        "maps",
        PythonExpression(expression=["'", "room_with_walls_keepout", "'", " + '.yaml'"])
    ])

    # List of lifecycle nodes for the keepout filter components
    lifecycle_nodes = ['keepout_filter_mask_server', 'keepout_costmap_filter_info_server']

    # Remappings for tf topics
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    # Value rewrites for the main params file (e.g., to enable/disable keepout zones)
    yaml_substitutions = {
        'KEEPOUT_ZONE_ENABLED': use_keepout_zones,   # passes the LaunchConfiguration
    }
   
    # Load the main kaapout zones parameters file, applying value rewrites and namespace prefix
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=yaml_substitutions,
        convert_types=True)

    # Environment variable to fix log buffering
    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1'
    )

    # Declare launch arguments that can be overridden from the command line
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace', default_value='', description='Top-level namespace'
    )

    declare_keepout_mask_yaml_cmd = DeclareLaunchArgument(
        'keepout_mask_yaml_file',
        default_value= keepout_mask_path,
        description='Full path to speed mask yaml file to load',
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='True', description='Use simulation (Gazebo) clock'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(navigation_pkg_path, 'config', 'keepout_filter_params.yaml'),
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

    declare_use_keepout_zones_cmd = DeclareLaunchArgument(
        'use_keepout_zones', default_value='True', description='Enable or disable keepout zones'
    )

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
                condition=IfCondition(use_keepout_zones),
                package='nav2_map_server',
                executable='map_server',
                name='keepout_filter_mask_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params, {'yaml_filename': keepout_mask_yaml_file}],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            ),
            # Filter info server that converts the mask into a costmap filter
            Node(
                condition=IfCondition(use_keepout_zones),
                package='nav2_map_server',
                executable='costmap_filter_info_server',
                name='keepout_costmap_filter_info_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            ),
            # Lifecycle manager for the two keepout components
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_keepout_zone',
                output='screen',
                arguments=['--ros-args', '--log-level', log_level],
                parameters=[{'autostart': autostart, 'use_sim_time': use_sim_time,}, {'node_names': lifecycle_nodes}],
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
            # Load the map server and filter info server as composable nodes
            LoadComposableNodes(
                target_container=container_name_full,
                condition=IfCondition(use_keepout_zones),
                composable_node_descriptions=[
                    ComposableNode(
                        package='nav2_map_server',
                        plugin='nav2_map_server::MapServer',
                        name='keepout_filter_mask_server',
                        parameters=[configured_params, {'yaml_filename': keepout_mask_yaml_file}],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package='nav2_map_server',
                        plugin='nav2_map_server::CostmapFilterInfoServer',
                        name='keepout_costmap_filter_info_server',
                        parameters=[{'use_sim_time': use_sim_time}],
                        remappings=remappings,
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
                        name='lifecycle_manager_keepout_zone',
                        parameters=[{'autostart': autostart, 'node_names': lifecycle_nodes}],
                    ),
                ],
            ),
        ],
    )

    # ----- Build the launch description -----
    ld = LaunchDescription()

    # Environment
    ld.add_action(stdout_linebuf_envvar)

    # Declared arguments
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_keepout_mask_yaml_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_container_name_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_use_keepout_zones_cmd)
    ld.add_action(declare_log_level_cmd)

    # Actions
    ld.add_action(load_nodes)
    ld.add_action(load_composable_nodes)

    return ld

