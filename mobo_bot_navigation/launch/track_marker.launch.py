from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch.actions import DeclareLaunchArgument
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution

def generate_launch_description():


    apriltag_ros_config_filename = 'tags_36h11.yaml'

    # Set the path to different files and folders
    pkg_share_docking = get_package_share_directory('mobo_bot_navigation')

    default_apriltag_ros_config_file_path = PathJoinSubstitution(
        [pkg_share_docking, 'config', apriltag_ros_config_filename])
    
      # Launch configuration variables
    apriltag_config_file = LaunchConfiguration('apriltag_config_file')
    tag_family = LaunchConfiguration('tag_family')
    tag_id = LaunchConfiguration('tag_id')
    use_sim_time = LaunchConfiguration('use_sim_time')
    camera_frame = LaunchConfiguration('camera_frame')


    declare_apriltag_config_file_cmd = DeclareLaunchArgument(
        name='apriltag_config_file',
        default_value=default_apriltag_ros_config_file_path,
        description='Full path to the AprilTag config file to use')
    
    declare_camera_frame_cmd = DeclareLaunchArgument(
        name='camera_frame',
        default_value='camera_optical',
        description='Frame for the camera'
    )

    declare_tag_family_cmd = DeclareLaunchArgument(
        name='tag_family',
        default_value='tag36h11',
        description='Family of AprilTag being used'
    )

    declare_tag_id_cmd = DeclareLaunchArgument(
        name='tag_id',
        default_value='0',
        description='ID of the AprilTag being used'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )



    composable_nodes = [
        ComposableNode(
            package='image_proc',
            plugin='image_proc::RectifyNode',
            name='rectify_node',
            remappings=[
                ('image', '/camera_optical/image_raw'),
                ('camera_info', '/camera_optical/camera_info'), 
                #('image_rect', 'image_rect'),
            ],
             parameters=[{
            'queue_size': 5,
            'interpolation': 1,
            'use_sim_time': use_sim_time,
        }],
        extra_arguments=[{'use_intra_process_comms': True}]
        ),
        
        ComposableNode(
            package='apriltag_ros',
            plugin='AprilTagNode',
            name='apriltag_node',
            parameters=[  
                apriltag_config_file,
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('image_rect', '/image_rect'),
                ('camera_info', '/camera_optical/camera_info'),
            ],
            extra_arguments=[{'use_intra_process_comms': True}]

        )
    ]


    container = ComposableNodeContainer(
        name='image_proc_container',
        namespace='', # Or your namespace
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
    )


    # Create the detected dock pose publisher node
    start_detected_dock_pose_publisher = Node(
        package='mobo_bot_navigation',
        executable='detected_dock_pose_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'parent_frame': [camera_frame],
            'child_frame': [tag_family, TextSubstitution(text=':'), tag_id],
            'publish_rate': 10.0
        }],
        arguments=['--ros-args', '--log-level', 'info'],
        output='screen'
    )

      # Add the arguments
    ld = LaunchDescription()
    ld.add_action(declare_apriltag_config_file_cmd)
    ld.add_action(declare_tag_family_cmd)
    ld.add_action(declare_tag_id_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_camera_frame_cmd)
    ld.add_action(start_detected_dock_pose_publisher)
    ld.add_action(container)

    return ld