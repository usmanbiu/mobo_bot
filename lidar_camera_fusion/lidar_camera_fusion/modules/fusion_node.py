#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2, CameraInfo
from rtabmap_ros.msg import KeyPoint
from cv_bridge import CvBridge
import tf2_ros
import numpy as np

from modules.reflectivity_processor import ReflectivityProcessor
from modules.orb_extractor import ORBExtractor
from modules.sensor_fusion import SensorFusion

class LidarCameraFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion_node')
        
        # Initialize modules
        self.reflectivity_processor = ReflectivityProcessor()
        self.orb_extractor = ORBExtractor(n_features=100)
        self.sensor_fusion = SensorFusion()
        
        # ROS 2 infrastructure
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None
        
        # Publishers
        self.pub_orb = self.create_publisher(KeyPoint, '/orb_features', 10)
        self.pub_reflectivity = self.create_publisher(PointCloud2, '/reflectivity_landmarks', 10)
        self.pub_debug = self.create_publisher(Image, '/fusion_debug', 10)
        
        # Subscribers
        self.sub_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_camera = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
        self.get_logger().info("Lidar-Camera Fusion Node Started")

    def camera_info_callback(self, msg):
        """Store camera calibration parameters."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.camera_frame = msg.header.frame_id
        self.get_logger().info("Camera calibration received")

    def lidar_callback(self, msg):
        """Process lidar data and publish reflectivity landmarks."""
        try:
            landmarks, debug_info = self.reflectivity_processor.process_scan(
                msg.ranges, msg.intensities, msg.angle_min, msg.angle_max, 
                msg.range_min, msg.range_max
            )
            
            # Publish landmarks
            cloud_msg = self.reflectivity_processor.landmarks_to_pointcloud2(landmarks, msg.header)
            if cloud_msg:
                self.pub_reflectivity.publish(cloud_msg)
                
            self.get_logger().info(
                f"Lidar: {debug_info['valid_points']} valid, "
                f"{debug_info['high_reflectivity']} reflective, "
                f"{debug_info['clusters']} clusters", 
                throttle_duration_sec=2.0
            )
                
        except Exception as e:
            self.get_logger().error(f"Lidar processing failed: {str(e)}")

    def image_callback(self, msg):
        """Process camera image and fuse with lidar data."""
        try:
            # Extract ORB features
            keypoints, descriptors, debug_image = self.orb_extractor.extract_features(msg)
            
            # Publish ORB features
            orb_msg = self.orb_extractor.features_to_keypoint_msg(keypoints, descriptors, msg.header)
            self.pub_orb.publish(orb_msg)
            
            # Perform sensor fusion if we have camera calibration
            if (self.camera_matrix is not None and 
                hasattr(self, 'current_landmarks') and 
                self.current_landmarks):
                
                self.fuse_sensors(keypoints, descriptors, debug_image, msg.header)
                
            self.get_logger().info(
                f"Camera: {len(keypoints)} ORB features", 
                throttle_duration_sec=2.0
            )
                
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {str(e)}")

    def fuse_sensors(self, keypoints, descriptors, debug_image, header):
        """Fuse lidar landmarks with camera features."""
        try:
            # Get transform from lidar to camera
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame, 'lidar_frame', rclpy.time.Time()
            )
            
            # Project landmarks to camera
            projected_points = self.sensor_fusion.project_lidar_to_camera(
                self.current_landmarks, transform, self.camera_matrix, self.dist_coeffs
            )
            
            # Associate with ORB features
            image_shape = debug_image.shape[:2]
            associations, _ = self.sensor_fusion.associate_features(
                projected_points, keypoints, descriptors, image_shape
            )
            
            # Create hybrid descriptors
            hybrid_descriptors, fusion_stats = self.sensor_fusion.create_hybrid_descriptors(associations)
            
            # Visualize associations on debug image
            for u, v, intensity, landmark in projected_points:
                if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:
                    cv2.circle(debug_image, (u, v), 6, (255, 0, 0), -1)
                    cv2.putText(debug_image, f"{intensity:.2f}", (u+10, v), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Publish debug image
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = header
            self.pub_debug.publish(debug_msg)
            
            self.get_logger().info(
                f"Fusion: {fusion_stats['successful_fusions']}/{fusion_stats['total_associations']} hybrid descriptors", 
                throttle_duration_sec=2.0
            )
            
        except tf2_ros.LookupException:
            self.get_logger().warn("TF lookup failed - check calibration")
        except Exception as e:
            self.get_logger().error(f"Sensor fusion failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()