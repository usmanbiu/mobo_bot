# Example:
#
#   Bringup turtlebot3:
#     $ export TURTLEBOT3_MODEL=waffle
#     $ export LDS_MODEL=LDS-01
#     $ ros2 launch turtlebot3_bringup robot.launch.py
#
#   SLAM:
#     $ ros2 launch rtabmap_demos turtlebot3_scan.launch.py
#
#   Navigation (install nav2_bringup package):
#     $ ros2 launch nav2_bringup navigation_launch.py
#     $ ros2 launch nav2_bringup rviz_launch.py
#
#   Teleop:
#     $ ros2 run turtlebot3_teleop teleop_keyboard

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time') 
    localization = LaunchConfiguration('localization').perform(context)
    localization = localization == 'True' or localization == 'true'
    icp_odometry = LaunchConfiguration('icp_odometry').perform(context)
    icp_odometry = icp_odometry == 'True' or icp_odometry == 'true'
    initial_pose = LaunchConfiguration('initial_pose').perform(context)
    
    parameters={
          'frame_id':'base_link',
          'odom_frame_id':'odom',
          'map_frame_id': 'map',
          'publish_tf': True,
          'use_sim_time':use_sim_time,
          'subscribe_depth':False,
          'subscribe_rgb':True,
          'subscribe_scan':True,
          'approx_sync':True,
          'use_action_for_goal':True,
          'Reg/Strategy':'1',
          'Reg/Force3DoF':'true',
          'RGBD/NeighborLinkRefining':'True',
          #'RGBD/StartAtOrigin':'True',
          'Grid/RangeMin':'0.2', # ignore laser scan points on the robot itself
          'Optimizer/GravitySigma':'0', # Disable imu constraints (we are already in 2D)
          'Vis/FeatureType':'2',               # ORB features for visual odometery
          'Kp/DetectorStrategy': '2',          # ORB features for loop closure detection
          'Kp/NNStrategy': '3',                #bruteforce matching
          'Kp/NndrRatio': '0.8',               # The Nearest Neighbor Distance Ratio. 
          'sync_queue_size':30,
          'topic_queue_size':30,
          #'RGBD/ProximityPathMaxNeighbors': '0',
          #'Mem/StereoFromMotion': 'true',
          #'Vis/EstimationType': '2',

    }
    arguments = []
    if localization:
        parameters['Mem/IncrementalMemory'] = 'False'
        parameters['Mem/InitWMWithAllNodes'] = 'True'
    else:
        arguments.append('-d') # This will delete the previous database (~/.ros/rtabmap.db)
               
                #change to /camera_optical/image_converted if your camera supports a
               #rtabmap acceptable format by default, works with the camera converter node below, also
               #comment out that node

    if initial_pose and initial_pose != '':
        arguments.extend(['--initial_pose', initial_pose])

    remappings=[
          ('scan','/lidar/scan'),
          ('rgb/image','/camera/image_augmented'), #camera/image_augmented  /camera_optical/image
          ('rgb/camera_info','/camera_optical/camera_info')] 
    if icp_odometry:
        remappings.append(('odom', 'icp_odom'))
    
    return [
        # Nodes to launch
        #used this node to convert from YUYV format to bgr8, comment this node if your camera supports bgr8 
        # or an rtabmap acceptable format by default
        # Node(
        #     package='camera_converter',
        #     executable='cam_formater',
        #     #name='camera_format_converter',
        #     output='screen'
        # ),
        
        # ICP odometry (optional)
        Node(
            condition=IfCondition(LaunchConfiguration('icp_odometry')),
            package='rtabmap_odom', executable='icp_odometry', output='screen',
            parameters=[parameters, 
                        {'odom_frame_id':'icp_odom',
                         'guess_frame_id':'odom'}],
            remappings=remappings),
        
        # SLAM:
        Node(
            package='rtabmap_slam', executable='rtabmap', output='screen',
            parameters=[parameters],
            remappings=remappings,
            arguments=arguments),

        # Visualization
        Node(
            package='rtabmap_viz', executable='rtabmap_viz', output='screen',
            parameters=[parameters],
            remappings=remappings,
            arguments=['-d', '/home/usman/.ros/rtabmapGUI.ini']),
     ]

def generate_launch_description():
    return LaunchDescription([

        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        
        DeclareLaunchArgument(
            'localization', default_value='false',
            description='Launch in localization mode.'),
        
        DeclareLaunchArgument(
            'icp_odometry', default_value='false',
            description='Launch ICP odometry on top of wheel odometry.'),

        DeclareLaunchArgument('initial_pose', default_value='',
            description='Initial pose for robot: "x y z roll pitch yaw"'),

        OpaqueFunction(function=launch_setup),
    ])




