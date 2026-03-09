#!/usr/bin/env python3
import numpy as np
from sklearn.cluster import DBSCAN
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
import struct
from std_msgs.msg import Header
from rclpy.node import Node
import rclpy

class ReflectivityProcessor:
    """Processes RPLIDAR C1 reflectivity data to create semantic landmarks."""
    
    def __init__(self, 
                 reflectivity_threshold=0.6,
                 dbscan_eps=0.3,
                 dbscan_min_samples=3,
                 max_landmarks=50):
        
        self.reflectivity_threshold = reflectivity_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.max_landmarks = max_landmarks
        self.landmark_id_counter = 0
        
    def extract_high_reflectivity_points(self, scan_msg):
        """
        Extract points with high reflectivity from LaserScan.
        
        Args:
            scan_msg: sensor_msgs/LaserScan message
            
        Returns:
            points_2d: List of (x, y, intensity) in lidar frame
        """
        ranges = np.array(scan_msg.ranges)
        intensities = np.array(scan_msg.intensities)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
        # Filter valid points
        valid_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        valid_ranges = ranges[valid_mask]
        valid_intensities = intensities[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return []
        
        # Convert to Cartesian
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        # Filter high reflectivity points
        if np.max(valid_intensities) > 0:
            threshold = self.reflectivity_threshold * np.max(valid_intensities)
            high_reflectivity_mask = valid_intensities > threshold
        else:
            high_reflectivity_mask = valid_intensities > 0.2
            
        high_reflectivity_points = np.column_stack([
            x[high_reflectivity_mask],
            y[high_reflectivity_mask],
            valid_intensities[high_reflectivity_mask]
        ])
        
        return high_reflectivity_points.tolist()
    
    def cluster_landmarks(self, points_2d):
        """
        Cluster reflectivity points into landmarks.
        
        Args:
            points_2d: List of (x, y, intensity)
            
        Returns:
            landmarks: List of [x, y, avg_intensity, landmark_id]
        """
        if len(points_2d) < self.dbscan_min_samples:
            return []
        
        points_array = np.array(points_2d)
        
        # Cluster based on position only (first 2 columns)
        clustering = DBSCAN(eps=self.dbscan_eps, 
                           min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(points_array[:, :2])
        
        landmarks = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
                
            cluster_points = points_array[labels == label]
            
            # Calculate cluster center and average intensity
            center_x = np.mean(cluster_points[:, 0])
            center_y = np.mean(cluster_points[:, 1])
            avg_intensity = np.mean(cluster_points[:, 2])
            num_points = len(cluster_points)
            
            # Only keep significant clusters
            if num_points >= self.dbscan_min_samples:
                landmark_id = self.landmark_id_counter
                self.landmark_id_counter += 1
                
                landmarks.append({
                    'id': landmark_id,
                    'x': float(center_x),
                    'y': float(center_y),
                    'z': 0.0,  # 2D lidar
                    'intensity': float(avg_intensity),
                    'num_points': num_points,
                    'covariance': np.cov(cluster_points[:, :2], rowvar=False).flatten().tolist()
                })
        
        # Limit number of landmarks
        if len(landmarks) > self.max_landmarks:
            landmarks = sorted(landmarks, key=lambda x: x['intensity'], reverse=True)[:self.max_landmarks]
        
        return landmarks
    
    def landmarks_to_pointcloud(self, landmarks, frame_id="lidar_frame"):
        """
        Convert landmarks to PointCloud2 message.
        
        Args:
            landmarks: List of landmark dictionaries
            frame_id: TF frame ID
            
        Returns:
            PointCloud2 message with fields: x, y, z, intensity, landmark_id
        """
        if not landmarks:
            return None
        
        cloud_msg = PointCloud2()
        cloud_msg.header = Header()
        cloud_msg.header.frame_id = frame_id
        cloud_msg.header.stamp = Node.get_clock().now().to_msg()
        
        # Define custom fields including landmark_id
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='landmark_id', offset=16, datatype=PointField.UINT32, count=1)
        ]
        
        cloud_msg.point_step = 20  # 5 fields * 4 bytes each
        cloud_msg.row_step = cloud_msg.point_step * len(landmarks)
        cloud_msg.is_dense = True
        cloud_msg.height = 1
        cloud_msg.width = len(landmarks)
        
        # Pack data
        cloud_data = bytearray()
        for landmark in landmarks:
            cloud_data.extend(struct.pack('ffffI', 
                landmark['x'], landmark['y'], landmark['z'],
                landmark['intensity'], landmark['id']))
        
        cloud_msg.data = bytes(cloud_data)
        return cloud_msg
    
    def create_reflectivity_descriptors(self, landmarks):
        """
        Create descriptors from reflectivity landmarks for loop closure.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            descriptors: List of descriptor vectors for each landmark
        """
        descriptors = []
        
        for landmark in landmarks:
            # Create a simple descriptor based on intensity and position
            # This can be expanded based on your needs
            descriptor = [
                landmark['intensity'],  # Primary feature
                landmark['num_points'] / 10.0,  # Normalized cluster size
                np.sqrt(landmark['x']**2 + landmark['y']**2) / 10.0,  # Normalized distance
                landmark['x'] / landmark['y'] if landmark['y'] != 0 else 0,  # Angular feature
            ]
            descriptors.append(descriptor)
        
        return descriptors
    













#     #!/usr/bin/env python3
# # import rclpy
# # from rclpy.node import Node
# # from sensor_msgs.msg import Image, PointCloud2, CameraInfo
# # from geometry_msgs.msg import PointStamped
# # from cv_bridge import CvBridge
# # import cv2
# # import numpy as np
# # import tf2_ros
# # from tf2_geometry_msgs import do_transform_point

# class VisualAugmentor(Node):
#     """
#     Augments camera images with synthetic markers at reflectivity landmark locations.
#     This makes reflectivity landmarks VISUALLY distinctive for ORB feature extraction.
#     """
    
#     def __init__(self):
#         super().__init__('svisual_augmentor')
        
#         # Subscribers
#         self.sub_image = self.create_subscription(
#             Image, '/camera/image_raw', self.image_callback, 10)
        
#         self.sub_scan = self.create_subscription(
#             LaserScan, 'lidar/scan', self.scan_callback, 10)
        
#         self.sub_camera_info = self.create_subscription(
#             CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
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
        
#         # Camera parameters
#         self.K = None
#         self.D = None
#         self.camera_frame = None
        
#         # Current state
#         self.current_landmarks = []  # List of (x, y, z, intensity) in lidar frame
#         self.marker_db = {}  # Stores marker patterns for consistency
        
#         # Marker generation parameters
#         self.marker_size = 40  # pixels
#         self.marker_opacity = 0.6  # Blend with original image
#         self.min_landmark_intensity = 0.7  # Only augment strong reflectivity
        
#         self.get_logger().info("Visual Augmentor Initialized")
    
#     def camera_info_callback(self, msg):
#         """Store camera calibration parameters."""
#         self.K = np.array(msg.k).reshape(3, 3)
#         self.D = np.array(msg.d)
#         self.camera_frame = msg.header.frame_id
    
#     def scan_callback(self, msg):
#         """Process incoming laser scan for RTAB-Map."""
#         # 1. Publish raw scan (for RTAB-Map's odometry)
#         raw_msg = LaserScan()
#         raw_msg.header = msg.header
#         raw_msg.header.frame_id = self.frame_id
#         raw_msg.angle_min = msg.angle_min
#         raw_msg.angle_max = msg.angle_max
#         raw_msg.angle_increment = msg.angle_increment
#         raw_msg.time_increment = msg.time_increment
#         raw_msg.scan_time = msg.scan_time
#         raw_msg.range_min = msg.range_min
#         raw_msg.range_max = msg.range_max
#         raw_msg.ranges = msg.ranges
#         raw_msg.intensities = msg.intensities
#         #self.pub_scan_raw.publish(raw_msg)
        
#         # 2. Extract high reflectivity points
#         high_reflectivity_indices = self.extract_high_reflectivity(msg)
        
#         # 3. Create processed scan with reflectivity markers
#         # processed_msg = self.create_processed_scan(msg, high_reflectivity_indices)
#         # self.pub_scan_processed.publish(processed_msg)
        
#         # 4. Create point cloud of landmarks
#         # landmark_cloud = self.create_landmark_cloud(msg, high_reflectivity_indices)
#         # if landmark_cloud:
#         #     self.pub_cloud_landmarks.publish(landmark_cloud)
        
#         # 5. Create custom descriptors
#         # descriptors = self.create_reflectivity_descriptors(msg, high_reflectivity_indices)
#         # if descriptors:
#         #     self.pub_descriptors.publish(descriptors)
    
#     def extract_high_reflectivity(self, scan_msg):
#         """Extract indices of high reflectivity points."""
#         intensities = np.array(scan_msg.intensities)
        
#         if len(intensities) == 0:
#             return []
        
#         # Calculate threshold
#         max_intensity = np.max(intensities)
#         if max_intensity > 0:
#             threshold = self.reflectivity_threshold * max_intensity
#         else:
#             threshold = 0.2
        
#         # Find high reflectivity points
#         high_indices = np.where(intensities > threshold)[0]
        
#         # Filter by valid range
#         ranges = np.array(scan_msg.ranges)
#         valid_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
#         high_indices = high_indices[valid_mask[high_indices]]
        
#         high_indices = high_indices.tolist()
    
#         if not high_indices:
#             return None
        
#         intensities = np.array(scan_msg.intensities)
#         angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
#         # Convert selected points to Cartesian
#         selected_ranges = ranges[high_indices]
#         selected_intensities = intensities[high_indices]
#         selected_angles = angles[high_indices]
        
#         # Convert to Cartesian (2D to 3D with z=0)
#         x = selected_ranges * np.cos(selected_angles)
#         y = selected_ranges * np.sin(selected_angles)
#         z = np.zeros_like(x)

#         self.landmarks = []
   
#         self.landmarks.append({
#             'x': x, 'y': y, 'z': z,
#             'intensity': selected_intensities
#         })



#     def image_callback(self, msg):
#         """Augment image with markers at reflectivity landmark locations."""
#         if self.K is None or not self.landmarks:
#             # Pass through if no calibration or landmarks
#             self.pub_augmented.publish(msg)
#             return
        
#         try:
#             # Convert to OpenCV
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             augmented = cv_image.copy()
#             debug = cv_image.copy()
            
#             # Get transform from lidar to camera, put correct lidar frame
#             transform = self.tf_buffer.lookup_transform(
#                 self.camera_frame, 'lidar_frame', rclpy.time.Time())
            
#             # Process each landmark
#             markers_added = 0
#             for landmark in self.landmarks:
#                 # Project landmark to image
#                 uv = self.project_to_image(landmark, transform)
#                 if uv is None:
#                     continue
                
#                 u, v = uv
                
#                 # Check if within image bounds
#                 if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
#                     # Create or retrieve marker for this landmark
#                     marker_id = self.get_marker_id(landmark)
#                     marker_pattern = self.get_marker_pattern(marker_id)
                    
#                     # Blend marker onto image
#                     self.blend_marker(augmented, u, v, marker_pattern)
                    
#                     # Draw debug visualization
#                     cv2.circle(debug, (u, v), 10, (0, 255, 255), 2)
#                     cv2.putText(debug, f"{landmark['intensity']:.2f}", 
#                                (u+15, v), cv2.FONT_HERSHEY_SIMPLEX, 
#                                0.5, (0, 255, 255), 1)
                    
#                     markers_added += 1
            
#             # Publish augmented image
#             augmented_msg = self.bridge.cv2_to_imgmsg(augmented, encoding='bgr8')
#             augmented_msg.header = msg.header
#             self.pub_augmented.publish(augmented_msg)
            
#             # Publish debug visualization
#             debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
#             debug_msg.header = msg.header
#             self.pub_debug.publish(debug_msg)
            
#             self.get_logger().debug(
#                 f"Added {markers_added} synthetic markers",
#                 throttle_duration_sec=1.0
#             )
            
#         except tf2_ros.LookupException:
#             self.get_logger().warn("TF lookup failed - passing through image")
#             self.pub_augmented.publish(msg)
#         except Exception as e:
#             self.get_logger().error(f"Augmentation failed: {e}")
#             self.pub_augmented.publish(msg)
    
#     def project_to_image(self, landmark, transform):
#         """Project 3D landmark to 2D image coordinates."""
#         try:
#             # Create 3D point in lidar frame
#             pt_lidar = PointStamped()
#             pt_lidar.point.x = self.landmarks['x']
#             pt_lidar.point.y = self.landmarks['y']
#             pt_lidar.point.z = self.landmarks['z']
            
#             # Transform to camera frame
#             pt_camera = do_transform_point(pt_lidar, transform)
            
#             # Project to 2D
#             point_3d = np.array([[pt_camera.point.x, pt_camera.point.y, pt_camera.point.z]])
#             uv, _ = cv2.projectPoints(point_3d, np.zeros(3), np.zeros(3), self.K, self.D)
            
#             u, v = uv[0][0].astype(int)
#             return (u, v)
            
#         except Exception as e:
#             self.get_logger().debug(f"Projection failed: {e}")
#             return None
    
#     def get_marker_id(self, self.landmarks):
#         """Generate consistent marker ID for a landmark."""
#         # Use quantized position as ID for consistency
#         grid_size = 0.1  # 10cm grid
#         x_idx = int(self.landmarks['x'] / grid_size)
#         y_idx = int(self.landmarks['y'] / grid_size)
#         intensity_idx = int(self.landmarks['intensity'] * 10)
        
#         return f"{x_idx}_{y_idx}_{intensity_idx}"
    
#     def get_marker_pattern(self, marker_id):
#         """Get or create marker pattern for a given ID."""
#         if marker_id not in self.marker_db:
#             # Generate new marker pattern
#             pattern = self.generate_marker_pattern(marker_id)
#             self.marker_db[marker_id] = pattern
            
#             # Keep DB size manageable
#             if len(self.marker_db) > 100:
#                 oldest = list(self.marker_db.keys())[0]
#                 del self.marker_db[oldest]
        
#         return self.marker_db[marker_id]
    
#     def generate_marker_pattern(self, marker_id):
#         """
#         Generate a distinctive marker pattern optimized for ORB detection.
        
#         ORB loves:
#         - High contrast corners
#         - Asymmetric patterns
#         - Multiple scales
#         - Binary intensity transitions
#         """
#         size = self.marker_size
#         pattern = np.zeros((size, size, 3), dtype=np.uint8)
        
#         # Use marker_id to seed deterministic but varied patterns
#         seed = hash(marker_id) % 10000
#         np.random.seed(seed)
        
#         # Method 1: Checkerboard (excellent for ORB corners)
#         if np.random.rand() < 0.4:
#             cell_size = max(4, size // np.random.randint(4, 8))
#             for i in range(0, size, cell_size):
#                 for j in range(0, size, cell_size):
#                     if (i//cell_size + j//cell_size) % 2 == 0:
#                         color = (255, 255, 255)  # White
#                     else:
#                         color = (0, 0, 0)  # Black
#                     pattern[i:i+cell_size, j:j+cell_size] = color
        
#         # Method 2: Concentric circles with spokes
#         elif np.random.rand() < 0.7:
#             # White background
#             pattern.fill(255)
#             center = size // 2
            
#             # Draw concentric circles
#             for r in range(5, size//2, 5):
#                 color = 0 if (r // 5) % 2 == 0 else 255
#                 cv2.circle(pattern, (center, center), r, (color, color, color), 2)
            
#             # Add radial lines (creates corners!)
#             for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
#                 x2 = center + int((size//2 - 2) * np.cos(angle))
#                 y2 = center + int((size//2 - 2) * np.sin(angle))
#                 cv2.line(pattern, (center, center), (x2, y2), (0, 0, 0), 2)
        
#         # Method 3: Binary code pattern (unique per marker)
#         else:
#             # Convert marker_id to binary visual pattern
#             binary_str = bin(abs(hash(marker_id)))[2:].zfill(16)
            
#             # Create 4x4 grid from binary string
#             grid_size = size // 4
#             for i in range(4):
#                 for j in range(4):
#                     idx = i * 4 + j
#                     if idx < len(binary_str) and binary_str[idx] == '1':
#                         color = (255, 255, 255)
#                     else:
#                         color = (0, 0, 0)
                    
#                     pattern[i*grid_size:(i+1)*grid_size, 
#                            j*grid_size:(j+1)*grid_size] = color
        
#         # Add subtle noise (helps with scale invariance)
#         noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
#         pattern = np.clip(pattern.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
#         return pattern
    
#     def blend_marker(self, image, center_u, center_v, marker):
#         """Blend marker pattern onto image at specified location."""
#         h, w = marker.shape[:2]
#         half_h, half_w = h // 2, w // 2
        
#         # Calculate ROI bounds
#         y1 = max(0, center_v - half_h)
#         y2 = min(image.shape[0], center_v + half_h)
#         x1 = max(0, center_u - half_w)
#         x2 = min(image.shape[1], center_u + half_w)
        
#         # Adjust marker if near edges
#         m_y1 = half_h - (center_v - y1)
#         m_y2 = half_h + (y2 - center_v)
#         m_x1 = half_w - (center_u - x1)
#         m_x2 = half_w + (x2 - center_u)
        
#         # Extract ROI and marker subregion
#         roi = image[y1:y2, x1:x2]
#         marker_roi = marker[m_y1:m_y2, m_x1:m_x2]
        
#         # Resize marker if needed (due to edge cropping)
#         if marker_roi.shape != roi.shape:
#             marker_roi = cv2.resize(marker_roi, (roi.shape[1], roi.shape[0]))
        
#         # Alpha blending
#         alpha = self.marker_opacity
#         blended = cv2.addWeighted(roi, 1-alpha, marker_roi, alpha, 0)
        
#         # Copy back to image
#         image[y1:y2, x1:x2] = blended

# def main(args=None):
#     rclpy.init(args=args)
#     node = VisualAugmentor()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()