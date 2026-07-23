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
    bringup_dir = get_package_share_directory('nav2_bringup')
    navigation_pkg_path = get_package_share_directory('mobo_bot_navigation')

    # Launch configurations (set via CLI or default values)
    namespace = LaunchConfiguration('namespace')
    speed_mask_yaml_file = LaunchConfiguration('speed_mask_yaml_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    use_respawn = LaunchConfiguration('use_respawn')
    use_speed_zones = LaunchConfiguration('use_speed_zones')
    log_level = LaunchConfiguration('log_level')

    # Fixed values (not user-configurable)
    autostart = True

    # Build the path to the speed mask YAML file (static inside your package)
    speed_mask_path = PathJoinSubstitution([
        navigation_pkg_path,
        "maps",
        PythonExpression(expression=["'", "mobo_map_speed", "'", " + '.yaml'"])
    ])

    lifecycle_nodes = ['speed_filter_mask_server', 'speed_costmap_filter_info_server']

    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    yaml_substitutions = {
        'SPEED_ZONE_ENABLED': use_speed_zones,
    }

   
    # Load the main Nav2 parameters file, applying value rewrites and namespace prefix
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=yaml_substitutions,
        convert_types=True)

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1'
    )

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace', default_value='', description='Top-level namespace'
    )

    declare_speed_mask_yaml_cmd = DeclareLaunchArgument(
        'speed_mask_yaml_file',
        default_value=speed_mask_path,
        description='Full path to speed mask yaml file to load',
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true',
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(navigation_pkg_path, 'config', 'speed_filter_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes',
    )

    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition',
        default_value='False',
        description='Use composed bringup if True',
    )

    declare_container_name_cmd = DeclareLaunchArgument(
        'container_name',
        default_value='nav2_container',
        description='the name of container that nodes will load in if use composition',
    )

    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn',
        default_value='False',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.',
    )

    declare_use_speed_zones_cmd = DeclareLaunchArgument(
        'use_speed_zones', default_value='True',
        description='Whether to enable speed zones or not'
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', default_value='info', description='log level'
    )


    load_nodes = GroupAction(
        condition=IfCondition(PythonExpression(['not ', use_composition])),
        actions=[
            PushROSNamespace(namespace),
            SetParameter('use_sim_time', use_sim_time),
            Node(
                condition=IfCondition(use_speed_zones),
                package='nav2_map_server',
                executable='map_server',
                name='speed_filter_mask_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params, {'yaml_filename': speed_mask_yaml_file}],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            ),
            Node(
                condition=IfCondition(use_speed_zones),
                package='nav2_map_server',
                executable='costmap_filter_info_server',
                name='speed_costmap_filter_info_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            ),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_speed_zone',
                output='screen',
                arguments=['--ros-args', '--log-level', log_level],
                parameters=[{'autostart': autostart}, {'node_names': lifecycle_nodes}],
            ),
        ],
    )

    # ----- Composed nodes (inside a container) -----
    container_name_full = (namespace, '/', container_name)


    # LoadComposableNode for map server twice depending if we should use the
    # value of map from a CLI or launch default or user defined value in the
    # yaml configuration file. They are separated since the conditions
    # currently only work on the LoadComposableNodes commands and not on the
    # ComposableNode node function itself
    load_composable_nodes = GroupAction(
        condition=IfCondition(use_composition),
        actions=[
            PushROSNamespace(namespace),
            SetParameter('use_sim_time', use_sim_time),
            LoadComposableNodes(
                target_container=container_name_full,
                condition=IfCondition(use_speed_zones),
                composable_node_descriptions=[
                    ComposableNode(
                        package='nav2_map_server',
                        plugin='nav2_map_server::MapServer',
                        name='speed_filter_mask_server',
                        parameters=[configured_params,
                            {'yaml_filename': speed_mask_yaml_file}
                        ],
                        remappings=remappings,
                    ),
                    ComposableNode(
                        package='nav2_map_server',
                        plugin='nav2_map_server::CostmapFilterInfoServer',
                        name='speed_costmap_filter_info_server',
                        parameters=[configured_params],
                        remappings=remappings,
                    ),
                ],
            ),

            LoadComposableNodes(
                target_container=container_name_full,
                composable_node_descriptions=[
                    ComposableNode(
                        package='nav2_lifecycle_manager',
                        plugin='nav2_lifecycle_manager::LifecycleManager',
                        name='lifecycle_manager_speed_zone',
                        parameters=[
                            {'autostart': autostart, 'node_names': lifecycle_nodes}
                        ],
                    ),
                ],
            ),
        ],
    )

 
    # Create the launch description and populate
    ld = LaunchDescription()

    # Set environment variables
    ld.add_action(stdout_linebuf_envvar)

    # Declare the launch options
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_speed_mask_yaml_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_container_name_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_use_speed_zones_cmd)
    ld.add_action(declare_log_level_cmd)

    # Add the actions to launch all of the map modifier nodes
    ld.add_action(load_nodes)
    ld.add_action(load_composable_nodes)

    return ld

