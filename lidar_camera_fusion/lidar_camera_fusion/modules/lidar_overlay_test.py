#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import hashlib
from message_filters import ApproximateTimeSynchronizer, Subscriber

class VisualAugmentor(Node):
    """
    Augments camera images with synthetic markers at reflectivity landmark locations.
    Processes LaserScan directly to extract high-reflectivity landmarks.
    
    IMPORTANT: Lidar intensities are 0-255, but typically <50 in your environment.
    We normalize to 0-1 for consistent processing.
    """
    
    def __init__(self):
        super().__init__('visual_augmentor')
        
        # Enable ALL debug logging
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # Create subscribers for synchronization
        self.image_sub = Subscriber(self, Image, '/camera_optical/image')
        self.scan_sub = Subscriber(self, LaserScan, '/lidar/scan')
        
        # Set up approximate time synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.scan_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.sub_camera_info = self.create_subscription(
            CameraInfo, '/camera_optical/camera_info', self.camera_info_callback, 10)
        
        # Publishers
        self.pub_augmented = self.create_publisher(
            Image, '/camera/image_augmented', 10)
        
        self.pub_debug = self.create_publisher(
            Image, '/augmentation/debug', 10)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # OpenCV
        self.bridge = CvBridge()
        
        # Camera parameters
        self.K = None
        self.D = None
        self.camera_frame = None
        self.image_width = None
        self.image_height = None
        
        # Coordinate frames
        self.lidar_frame = 'lidar'
        self.map_frame = 'map'
        
        # Lidar sensor height
        self.sensor_height = 0

        # Lidar field of view
        fov_deg = 140.0
        self.half_fov_rad = np.deg2rad(fov_deg / 2.0)

        # Intensity parameters for 0-255 range
        self.reflectivity_threshold = 20
        self.relative_threshold_multiplier = 1.2
        self.min_cluster_size = 3
        
        # Marker parameters
        self.marker_size = 40
        self.marker_opacity = 0.6
        
        # Marker database
        self.marker_db = {}
        
        # Statistics
        self.stats = {
            'sync_calls': 0,
            'sync_skipped_no_calib': 0,
            'sync_skipped_no_landmarks': 0,
            'scans_processed': 0,
            'total_points': 0,
            'high_reflectivity_points': 0,
            'landmarks_created': 0
        }
        
        self.get_logger().info("Visual Augmentor Initialized")
    
    def camera_info_callback(self, msg):
        """Store camera calibration parameters."""
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)
        self.camera_frame = "camera_optical"
        self.image_width = msg.width
        self.image_height = msg.height
        self.get_logger().info(f"Camera calibration received: {self.image_width}x{self.image_height}")
    
    def sync_callback(self, image_msg, scan_msg):
        """
        Synchronized callback that receives time-aligned image and laser scan.
        This is the ONLY processing pipeline.
        """
        self.stats['sync_calls'] += 1
        
        # Process scan to extract landmarks
        landmarks = self.extract_high_reflectivity_landmarks(scan_msg)
        
        # Log sync stats periodically
        if self.stats['sync_calls'] % 10 == 0:
            time_diff = abs(
                (image_msg.header.stamp.sec + image_msg.header.stamp.nanosec*1e-9) -
                (scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec*1e-9)
            )
            self.get_logger().info(
                f"Sync #{self.stats['sync_calls']}: Time diff={time_diff:.3f}s, "
                f"Landmarks={len(landmarks)}"
            )
        
        # Process the synchronized pair
        self.process_synchronized_data(image_msg, landmarks)
    
    def process_synchronized_data(self, image_msg, landmarks):
        """Process time-synchronized image and landmarks."""
        if self.K is None:
            self.stats['sync_skipped_no_calib'] += 1
            self.get_logger().warn("No camera calibration yet, skipping augmentation")
            self.pub_augmented.publish(image_msg)
            return
        
        if not landmarks:
            self.stats['sync_skipped_no_landmarks'] += 1
            self.pub_augmented.publish(image_msg)
            return
        
        try:
            # Convert to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            augmented = cv_image.copy()
            debug = cv_image.copy()
            
            # Process each landmark
            markers_added = 0
            valid_landmarks = []
            
            for landmark in landmarks:
                # Project landmark to image
                uv = self.project_to_image(landmark)
                if uv is None:
                    continue
                
                u, v = uv
                
                # Check if within image bounds
                if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
                    valid_landmarks.append({
                        'uv': (u, v),
                        'landmark': landmark,
                        'marker_id': self.get_marker_id(landmark)
                    })
            
            if not valid_landmarks:
                self.get_logger().debug("No landmarks projected to image")
                self.pub_augmented.publish(image_msg)
                return
            
            # Sort by normalized intensity (strongest first)
            valid_landmarks.sort(key=lambda x: x['landmark']['intensity_normalized'], reverse=True)
            
            # Limit number of markers to avoid clutter
            max_markers = min(10, len(valid_landmarks))
            
            for i in range(max_markers):
                data = valid_landmarks[i]
                u, v = data['uv']
                landmark = data['landmark']
                
                # Create or retrieve marker
                marker_pattern = self.get_marker_pattern(data['marker_id'])
                
                # Blend marker onto image
                self.blend_marker(augmented, u, v, marker_pattern)
                
                # Draw debug visualization with color based on intensity
                intensity_norm = landmark['intensity_normalized']
                color_intensity = int(intensity_norm * 255)
                
                # Color gradient: blue (low) -> green (medium) -> red (high)
                if intensity_norm < 0.33:
                    color = (255, int(color_intensity * 3), 0)
                elif intensity_norm < 0.66:
                    color = (255 - int(color_intensity * 1.5), 255, 0)
                else:
                    color = (0, 255 - int(color_intensity * 0.5), color_intensity)
                
                cv2.circle(debug, (u, v), 8, color, 2)
                cv2.putText(debug, f"{landmark['intensity']:.0f}", 
                           (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
                
                markers_added += 1
            
            # Publish augmented image
            augmented_msg = self.bridge.cv2_to_imgmsg(augmented, encoding='bgr8')
            augmented_msg.header = image_msg.header
            self.pub_augmented.publish(augmented_msg)
            
            # Publish debug visualization with statistics
            debug_stats = (
                f"Sync #{self.stats['sync_calls']}: {markers_added}/{len(valid_landmarks)} | "
                f"Landmarks: {len(landmarks)}"
            )
            cv2.putText(debug, debug_stats, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.pub_debug.publish(debug_msg)
            
            self.get_logger().debug(
                f"Added {markers_added} synthetic markers from {len(landmarks)} landmarks",
                throttle_duration_sec=1.0
            )
            
        except Exception as e:
            self.get_logger().error(f"Augmentation failed: {e}")
            self.pub_augmented.publish(image_msg)
    
    def extract_high_reflectivity_landmarks(self, scan_msg):
        """
        Extract and cluster high reflectivity points from LaserScan.
        Intensities are 0-255.
        
        Returns: List of dictionaries with keys:
            'x', 'y', 'z', 'intensity', 'cluster_size', 'intensity_normalized'
        """
        ranges = np.array(scan_msg.ranges)
        intensities = np.array(scan_msg.intensities, dtype=np.float32)
        
        self.stats['total_points'] += len(intensities)
        
        if len(intensities) == 0:
            return []
        
        # Create angle array
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        
        # Filter valid points
        valid_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        valid_mask &= np.isfinite(intensities)
        
        # FOV filtering
        rear_fov_mask = (
            (angles <= np.pi - self.half_fov_rad) |
            (angles >= -np.pi + self.half_fov_rad)
        )
        # valid_mask &= rear_fov_mask  # Uncomment if needed
        
        valid_ranges = ranges[valid_mask]
        valid_intensities = intensities[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return []
        
        # Calculate thresholds
        max_intensity = np.max(valid_intensities)
        min_intensity = np.min(valid_intensities)
        median_intensity = np.median(valid_intensities)
        
        absolute_threshold = self.reflectivity_threshold
        relative_threshold = median_intensity * self.relative_threshold_multiplier
        
        threshold = max(absolute_threshold, relative_threshold)
        
        self.get_logger().debug(
            f"Intensity thresholds: max={max_intensity:.1f}, "
            f"median={median_intensity:.1f}, "
            f"final={threshold:.1f}",
            throttle_duration_sec=2.0
        )
        
        # Find high reflectivity points
        high_reflectivity_mask = valid_intensities > threshold
        self.stats['high_reflectivity_points'] += np.sum(high_reflectivity_mask)
        
        high_ranges = valid_ranges[high_reflectivity_mask]
        high_intensities = valid_intensities[high_reflectivity_mask]
        high_angles = valid_angles[high_reflectivity_mask]
        
        if len(high_ranges) == 0:
            return []
        
        # Convert to Cartesian coordinates
        x = high_ranges * np.cos(high_angles)
        y = high_ranges * np.sin(high_angles)
        z = np.full_like(x, self.sensor_height)
        
        # Adaptive clustering
        landmarks = []
        processed = np.zeros(len(x), dtype=bool)
        
        for i in range(len(x)):
            if processed[i]:
                continue
            
            distance = np.sqrt(x[i]**2 + y[i]**2)
            cluster_radius = 0.1 + 0.05 * (distance / 5.0)
            
            distances = np.sqrt((x - x[i])**2 + (y - y[i])**2)
            cluster_mask = distances < cluster_radius
            
            cluster_size = np.sum(cluster_mask)
            if cluster_size >= self.min_cluster_size:
                cluster_x = np.mean(x[cluster_mask])
                cluster_y = np.mean(y[cluster_mask])
                cluster_z = np.mean(z[cluster_mask])
                cluster_intensity = np.mean(high_intensities[cluster_mask])
                
                intensity_normalized = min(cluster_intensity / 100.0, 1.0)
                
                landmarks.append({
                    'x': float(cluster_x),
                    'y': float(cluster_y),
                    'z': float(cluster_z),
                    'intensity': float(cluster_intensity),
                    'intensity_normalized': float(intensity_normalized),
                    'cluster_size': int(cluster_size),
                    'distance': float(distance)
                })
                
                processed[cluster_mask] = True
                self.stats['landmarks_created'] += 1
        
        return landmarks
    
    def project_to_image(self, landmark):
        """
        Project a world-anchored LiDAR landmark into camera image pixels.
        Pipeline: LiDAR → MAP → CAMERA_OPTICAL → IMAGE
        """
        if self.K is None:
            self.get_logger().warn("Camera intrinsics not available")
            return None

        try:
            # Get lidar to map transform
            transform_l_m = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.lidar_frame,
                rclpy.time.Time()
            )
            
            # Get map to camera transform
            transform_m_c = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.map_frame,
                rclpy.time.Time()
            )
            
            # Transform point from lidar to map to camera
            p_lidar = PointStamped()
            p_lidar.header.frame_id = self.lidar_frame
            p_lidar.point.x = landmark["x"]
            p_lidar.point.y = landmark["y"]
            p_lidar.point.z = landmark["z"]

            p_map = do_transform_point(p_lidar, transform_l_m)
            p_cam = do_transform_point(p_map, transform_m_c)

            X = p_cam.point.x
            Y = p_cam.point.y
            Z = p_cam.point.z

            if Z <= 0.01:
                return None

            # Camera projection
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]

            u = int(fx * (X / Z) + cx)
            v = int(fy * (Y / Z) + cy)

            # Image bounds check
            if 0 <= u < self.image_width and 0 <= v < self.image_height:
                return (u, v)
            else:
                return None

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF error during projection: {e}")
            return None
    
    def get_marker_id(self, landmark):
        """Generate unique marker ID from landmark coordinates."""
        marker_str = f"{landmark['x']:.3f}_{landmark['y']:.3f}_{landmark['z']:.3f}"
        hash_obj = hashlib.md5(marker_str.encode())
        return hash_obj.hexdigest()[:8]
    
    def get_marker_pattern(self, marker_id):
        """Retrieve or generate marker pattern."""
        if marker_id not in self.marker_db:
            # Generate a simple circle pattern as fallback
            # In practice, you'd load actual marker images
            pattern = np.zeros((self.marker_size, self.marker_size, 3), dtype=np.uint8)
            center = self.marker_size // 2
            cv2.circle(pattern, (center, center), center-2, (0, 255, 0), -1)
            cv2.putText(pattern, marker_id[:2], (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.marker_db[marker_id] = pattern
        
        return self.marker_db[marker_id]
    
    def blend_marker(self, image, u, v, marker):
        """Blend marker onto image at specified location."""
        h, w = marker.shape[:2]
        half = h // 2
        
        # Calculate ROI
        x1 = max(0, u - half)
        x2 = min(image.shape[1], u + half)
        y1 = max(0, v - half)
        y2 = min(image.shape[0], v + half)
        
        # Calculate marker ROI
        mx1 = half - (u - x1)
        mx2 = half + (x2 - u)
        my1 = half - (v - y1)
        my2 = half + (y2 - v)
        
        # Blend
        roi = image[y1:y2, x1:x2]
        marker_roi = marker[my1:my2, mx1:mx2]
        
        # Alpha blend
        blended = cv2.addWeighted(roi, 1 - self.marker_opacity, 
                                  marker_roi, self.marker_opacity, 0)
        image[y1:y2, x1:x2] = blended

def main(args=None):
    rclpy.init(args=args)
    node = VisualAugmentor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()