#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import message_filters
import yaml

class ReflectivityProjector(Node):
    def __init__(self):
        super().__init__('reflectivity_projector')
        
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # TF2 setup for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.cluster_marker_pub = self.create_publisher(
            MarkerArray, '/reflectivity_clusters_markers', 10)
        self.debug_image_pub = self.create_publisher(
            Image, '/reflectivity_debug_image', 10)
        self.cluster_points_pub = self.create_publisher(
            Marker, '/cluster_points_3d', 10)  # 3D visualization in RVIZ
        
        # Synchronized subscribers for LaserScan, Image, and CameraInfo
        scan_sub = message_filters.Subscriber(self, LaserScan, '/lidar/scan')
        image_sub = message_filters.Subscriber(self, Image, '/camera_optical/image_raw')
        camera_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera_optical/camera_info')
        
        # Synchronize topics with 0.1s tolerance
        self.sync = ApproximateTimeSynchronizer(
            [scan_sub, image_sub, camera_info_sub], 
            queue_size=10, 
            slop=0.1)
        self.sync.registerCallback(self.sync_callback)
        
        # Parameters (tunable)
        self.reflectivity_threshold = 0.7  # 70% of max
        self.cluster_eps = 0.2  # meters for DBSCAN
        self.min_cluster_points = 3
        self.projection_radius = 5  # pixels for visualization
        
        self.get_logger().info('Reflectivity Projector Node Initialized')

    def sync_callback(self, scan_msg, image_msg, camera_info_msg):
        """Main callback with synchronized data"""
        try:
            # Step 1: Extract and cluster reflectivity points
            clusters_3d = self.process_reflectivity(scan_msg)
            
            if not clusters_3d:
                return
            
            # Step 2: Transform clusters to camera frame
            clusters_camera = self.transform_to_camera(clusters_3d, scan_msg.header.frame_id)
            
            if not clusters_camera:
                return
            
            # Step 3: Project to image coordinates
            image_points = self.project_to_image(clusters_camera, camera_info_msg)
            
            # Step 4: Draw on image
            debug_image = self.draw_on_image(image_msg, image_points)
            
            # Step 5: Publish results
            self.publish_results(clusters_3d, debug_image, scan_msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error in sync_callback: {str(e)}')

    def process_reflectivity(self, scan_msg):
        """Extract and cluster high-reflectivity points from LaserScan"""
        ranges = np.array(scan_msg.ranges)
        intensities = np.array(scan_msg.intensities)
        
        # Filter invalid ranges
        valid_idx = np.isfinite(ranges) & (ranges > 0.1)
        ranges = ranges[valid_idx]
        intensities = intensities[valid_idx]
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))[valid_idx]
        
        if len(ranges) == 0:
            return []
        
        # Find high reflectivity points
        max_intensity = np.max(intensities)
        if max_intensity == 0:
            return []
        
        high_reflectivity = intensities > (self.reflectivity_threshold * max_intensity)
        
        if not np.any(high_reflectivity):
            return []
        
        # Convert to Cartesian coordinates (in lidar frame)
        points_2d = []
        for i, is_high in enumerate(high_reflectivity):
            if is_high:
                r = ranges[i]
                theta = angles[i]
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points_2d.append([x, y, intensities[i]])  # Store intensity as 3rd dim for clustering
        
        if len(points_2d) < self.min_cluster_points:
            return []
        
        points_2d = np.array(points_2d)
        
        # Cluster the points using DBSCAN
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.min_cluster_points)
        labels = clustering.fit_predict(points_2d[:, :2])  # Use only x,y for clustering
        
        # Extract cluster centers
        clusters = []
        for label in set(labels):
            if label == -1:  # Skip noise
                continue
            cluster_points = points_2d[labels == label]
            center = np.mean(cluster_points[:, :2], axis=0)
            # Add z=0 for 3D point (assuming 2D lidar)
            cluster_3d = [center[0], center[1], 0.0]
            clusters.append(cluster_3d)
        
        return clusters

    def transform_to_camera(self, clusters_3d, source_frame):
        """Transform cluster points from lidar frame to camera optical frame"""
        try:
            # Look up transform from lidar to camera_optical_frame
            transform = self.tf_buffer.lookup_transform(
                'camera_optical',  # target frame
                source_frame,             # source frame
                rclpy.time.Time())
            
            clusters_camera = []
            for point in clusters_3d:
                # Create PointStamped in source frame
                point_src = tf2_geometry_msgs.PointStamped()
                point_src.header.frame_id = source_frame
                point_src.point.x = float(point[0])
                point_src.point.y = float(point[1])
                point_src.point.z = float(point[2])
                
                # Transform to camera frame
                point_cam = tf2_geometry_msgs.do_transform_point(point_src, transform)
                clusters_camera.append([
                    point_cam.point.x,
                    point_cam.point.y,
                    point_cam.point.z
                ])
            
            return clusters_camera
            
        except TransformException as e:
            self.get_logger().warn(f'Transform failed: {str(e)}')
            return []

    def project_to_image(self, clusters_camera, camera_info_msg):
        """Project 3D points in camera frame to 2D image coordinates"""
        # Get camera intrinsic matrix from CameraInfo
        K = np.array(camera_info_msg.k).reshape(3, 3)
        
        image_points = []
        for point in clusters_camera:
            # Point in camera frame: (x, y, z) where:
            # x: right, y: down, z: forward (typical for camera_optical_frame)
            x, y, z = point
            
            if z <= 0:  # Point behind camera
                continue
            
            # Project using pinhole camera model
            u = (K[0, 0] * x / z) + K[0, 2]
            v = (K[1, 1] * y / z) + K[1, 2]
            
            # Check if within image bounds (will check later when we have image size)
            image_points.append((int(u), int(v), z))  # Store z for depth info
        
        return image_points

    def draw_on_image(self, image_msg, image_points):
        """Draw projected cluster centers on the camera image"""
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        
        # Get image dimensions
        height, width = cv_image.shape[:2]
        
        # Draw each projected point
        for u, v, z in image_points:
            # Check if within image bounds
            if 0 <= u < width and 0 <= v < height:
                # Color based on distance (closer = red, farther = blue)
                color_intensity = min(255, int(255 * (5.0 / max(z, 0.5))))  # Normalize to ~5m
                color = (0, 255 - color_intensity, color_intensity)  # BGR format
                
                # Draw circle with radius based on distance? Or fixed size
                cv2.circle(cv_image, (u, v), self.projection_radius, color, -1)
                
                # Draw border
                cv2.circle(cv_image, (u, v), self.projection_radius, (255, 255, 255), 2)
                
                # Add distance text
                cv2.putText(cv_image, f'{z:.1f}m', (u+10, v-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert back to ROS Image
        return self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

    def publish_results(self, clusters_3d, debug_image, header):
        """Publish markers for RVIZ and debug image"""
        # Publish debug image
        self.debug_image_pub.publish(debug_image)
        
        # Create marker array for RVIZ visualization
        marker_array = MarkerArray()
        
        # Individual cluster markers (spheres)
        for i, point in enumerate(clusters_3d):
            marker = Marker()
            marker.header = header
            marker.header.frame_id = 'laser_frame'  # or appropriate frame
            marker.ns = 'reflectivity_clusters'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = float(point[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        
        self.cluster_marker_pub.publish(marker_array)
        
        # Also publish as point cloud alternative
        if clusters_3d:
            points_marker = Marker()
            points_marker.header = header
            points_marker.header.frame_id = 'laser_frame'
            points_marker.ns = 'cluster_points'
            points_marker.id = 0
            points_marker.type = Marker.POINTS
            points_marker.action = Marker.ADD
            points_marker.scale.x = 0.05
            points_marker.scale.y = 0.05
            points_marker.color.a = 1.0
            points_marker.color.r = 1.0
            points_marker.color.g = 0.0
            points_marker.color.b = 0.0
            
            for point in clusters_3d:
                p = Point()
                p.x = float(point[0])
                p.y = float(point[1])
                p.z = float(point[2])
                points_marker.points.append(p)
            
            self.cluster_points_pub.publish(points_marker)

def main(args=None):
    rclpy.init(args=args)
    node = ReflectivityProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# # Modified version with extensive debugging
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, LaserScan, CameraInfo
# from geometry_msgs.msg import PointStamped
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import tf2_ros
# from tf2_geometry_msgs import do_transform_point

# class VisualAugmentorDebug(Node):
#     def __init__(self):
#         super().__init__('visual_augmentor_debug')
        
#         # Enable ALL debug logging
#         self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
#         # Subscribers
#         self.sub_scan = self.create_subscription(
#             LaserScan, '/lidar/scan', self.scan_callback, 10)
        
#         self.sub_image = self.create_subscription(
#             Image, '/camera_optical/image_raw', self.image_callback, 10)
        
#         self.sub_camera_info = self.create_subscription(
#             CameraInfo, '/camera_optical/camera_info', self.camera_info_callback, 10)
        
#         # Publishers
#         self.pub_augmented = self.create_publisher(
#             Image, '/camera/image_augmented', 10)
        
#         self.pub_debug = self.create_publisher(
#             Image, '/augmentation/debug', 10)
        
#         # TF
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
#         # OpenCV
#         self.bridge = CvBridge()
        
#         # LOWER THRESHOLDS FOR TESTING
#         self.reflectivity_threshold = 0.3  # Changed from 0.7
#         self.min_intensity_absolute = 0.1  # Changed from 0.3
        
#         # Other params
#         self.marker_size = 40
#         self.marker_opacity = 0.6
#         self.camera_frame = None
#         self.lidar_frame = 'lidar'
#         self.K = None
#         self.D = None
        
#         # Debug counters
#         self.scan_count = 0
#         self.image_count = 0
#         self.landmarks_found = 0
        
#         self.get_logger().info("🔧 VISUAL AUGMENTOR DEBUG MODE STARTED")
    
#     def scan_callback(self, msg):
#         """Enhanced scan processing with detailed logging."""
#         self.scan_count += 1
        
#         # Log scan info
#         self.get_logger().debug(
#             f"Scan #{self.scan_count}: "
#             f"{len(msg.ranges)} points, "
#             f"Intensities: {'YES' if msg.intensities else 'NO'}"
#         )
        
#         if msg.intensities:
#             intensities = np.array(msg.intensities)
            
#             # Log intensity statistics
#             valid_intensities = intensities[intensities > 0]
#             if len(valid_intensities) > 0:
#                 self.get_logger().info(
#                     f"📊 Intensities - Min: {np.min(valid_intensities):.3f}, "
#                     f"Max: {np.max(valid_intensities):.3f}, "
#                     f"Mean: {np.mean(valid_intensities):.3f}"
#                 )
                
#                 # Check thresholds
#                 max_intensity = np.max(valid_intensities)
#                 threshold = max(self.reflectivity_threshold * max_intensity, 
#                               self.min_intensity_absolute)
                
#                 self.get_logger().info(
#                     f"🎯 Threshold calculation: "
#                     f"Max={max_intensity:.3f}, "
#                     f"Relative={self.reflectivity_threshold * max_intensity:.3f}, "
#                     f"Absolute={self.min_intensity_absolute:.3f}, "
#                     f"Using={threshold:.3f}"
#                 )
                
#                 # Count high reflectivity points
#                 high_count = np.sum(intensities > threshold)
#                 self.get_logger().info(f"🔦 High reflectivity points: {high_count}")
                
#                 if high_count > 0:
#                     # Extract and log some sample points
#                     high_indices = np.where(intensities > threshold)[0][:3]
#                     for idx in high_indices:
#                         angle = msg.angle_min + idx * msg.angle_increment
#                         self.get_logger().debug(
#                             f"  Point {idx}: angle={angle:.2f}rad, "
#                             f"range={msg.ranges[idx]:.2f}m, "
#                             f"intensity={intensities[idx]:.3f}"
#                         )
#             else:
#                 self.get_logger().warn("⚠️  No valid intensities > 0!")
#         else:
#             self.get_logger().warn("⚠️  No intensities array in scan!")
    
#     def camera_info_callback(self, msg):
#         self.K = np.array(msg.k).reshape(3, 3)
#         self.D = np.array(msg.d)
#         self.camera_frame = msg.header.frame_id
#         self.get_logger().info(f"📷 Camera calibration: frame={self.camera_frame}")
    
#     def image_callback(self, msg):
#         """Simple pass-through for now with logging."""
#         self.image_count += 1
        
#         self.get_logger().debug(
#             f"Image #{self.image_count} received"
#         )
        
#         # Just publish the original image for now
#         self.pub_augmented.publish(msg)
        
#         # Create simple debug image
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
#             # Add debug text
#             debug_image = cv_image.copy()
#             cv2.putText(debug_image, f"Scans: {self.scan_count}", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(debug_image, f"Images: {self.image_count}", 
#                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(debug_image, "DEBUG MODE", 
#                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
#             debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
#             debug_msg.header = msg.header
#             self.pub_debug.publish(debug_msg)
            
#         except Exception as e:
#             self.get_logger().error(f"Debug image creation failed: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = VisualAugmentorDebug()
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         print("\n👋 Debug augmentor stopped")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()








#     import numpy as np

# def project_lidar_to_image(
#     lidar_points,
#     R_lidar_to_cam,
#     t_lidar_to_cam,
#     camera_matrix
# ):
#     """
#     Projects LiDAR points into the camera image.

#     Parameters:
#     - lidar_points: (N, 3) array of LiDAR points [x, y, z]
#     - R_lidar_to_cam: (3, 3) rotation matrix
#     - t_lidar_to_cam: (3,) translation vector
#     - camera_matrix: (3, 3) intrinsic matrix

#     Returns:
#     - image_points: (N, 2) pixel coordinates [u, v]
#     - valid_mask: (N,) boolean mask (points in front of camera)
#     """

#     # Convert to numpy array
#     lidar_points = np.asarray(lidar_points)

#     # Transform LiDAR → Camera frame
#     cam_points = (R_lidar_to_cam @ lidar_points.T).T + t_lidar_to_cam

#     X = cam_points[:, 0]
#     Y = cam_points[:, 1]
#     Z = cam_points[:, 2]

#     # Keep points in front of the camera
#     valid_mask = Z > 0

#     X = X[valid_mask]
#     Y = Y[valid_mask]
#     Z = Z[valid_mask]

#     # Camera intrinsics
#     fx = camera_matrix[0, 0]
#     fy = camera_matrix[1, 1]
#     cx = camera_matrix[0, 2]
#     cy = camera_matrix[1, 2]

#     # Project to image plane
#     u = fx * (X / Z) + cx
#     v = fy * (Y / Z) + cy

#     image_points = np.stack([u, v], axis=1)

#     return image_points, valid_mask
