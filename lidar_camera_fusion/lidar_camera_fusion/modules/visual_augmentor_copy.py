#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import PointStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from tf_transformations import quaternion_matrix
from scipy.spatial.transform import Rotation
from tf2_ros import TransformRegistration




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

        # Subscribers
        self.sub_image = self.create_subscription(
            Image, '/camera_optical/image_raw', self.image_callback, 10)
        
        self.sub_scan = self.create_subscription(
            LaserScan, '/lidar/scan', self.scan_callback, 10)
        
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
        
        # Lidar parameters
        self.lidar_frame = 'lidar'  # Adjust based on your TF tree
        
        self.sensor_height = 0 #.15  #height of lidar scan from ground
        
        #map frame
        self.map_frame = 'map'

        #lidar field of view
        fov_deg = 140.0
        self.half_fov_rad = np.deg2rad(fov_deg / 2.0)  # ≈ 1.2217 rad


        # Current state
        self.current_landmarks = []  # List of dictionaries with x, y, z, intensity
        self.marker_db = {}  # Stores marker patterns for consistency
        
        # CRITICAL: Intensity parameters for 0-255 range (but typically <50)
        self.reflectivity_threshold = 20  # Absolute threshold in 0-255 range
        self.relative_threshold_multiplier = 1.2  # Times max intensity in scan
        self.min_cluster_size = 3  # Minimum points to form a landmark
        
        # Marker parameters
        self.marker_size = 40  # pixels
        self.marker_opacity = 0.6  # Blend with original image
        
        # Statistics for debugging
        self.stats = {
            'scans_processed': 0,
            'total_points': 0,
            'high_reflectivity_points': 0,
            'landmarks_created': 0
        }
        
        self.get_logger().info("Visual Augmentor Initialized for 0-255 intensity range")
        self.get_logger().info(f"Threshold: {self.reflectivity_threshold} (absolute), "
                              f"{self.relative_threshold_multiplier}x max (relative)")
    
    def camera_info_callback(self, msg):
        """Store camera calibration parameters and image dimensions."""
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)
        self.camera_frame = "camera_optical"  # msg.header.frame_id
        self.image_width = msg.width
        self.image_height = msg.height
        self.get_logger().info(f"Camera calibration received: {self.image_width}x{self.image_height}")
        
    def scan_callback(self, msg):
        """Process laser scan to extract reflectivity landmarks."""
        self.stats['scans_processed'] += 1
        
        try:
            landmarks = self.extract_high_reflectivity_landmarks(msg)
            
            # Update landmarks
            self.current_landmarks = landmarks
            
            # Log statistics periodically
            if self.stats['scans_processed'] % 10 == 0:
                self.log_intensity_stats(msg)
            
            self.get_logger().debug(
                f"Scan {self.stats['scans_processed']}: "
                f"Extracted {len(landmarks)} reflectivity landmarks",
                throttle_duration_sec=1.0
            )
            
        except Exception as e:
            self.get_logger().error(f"Scan processing failed: {e}")
    
    def log_intensity_stats(self, scan_msg):
        """Log intensity statistics for debugging."""
        intensities = np.array(scan_msg.intensities)
        
        if len(intensities) > 0:
            valid_intensities = intensities[intensities > 0]  # Filter zeros
            
            if len(valid_intensities) > 0:
                stats = {
                    'min': np.min(valid_intensities),
                    'max': np.max(valid_intensities),
                    'mean': np.mean(valid_intensities),
                    'median': np.median(valid_intensities),
                    'std': np.std(valid_intensities),
                    'above_20': np.sum(valid_intensities > 20),
                    'above_30': np.sum(valid_intensities > 30),
                    'above_50': np.sum(valid_intensities > 50),
                }
                
                self.get_logger().info(
                    f"Intensity stats: "
                    f"min={stats['min']:.1f}, "
                    f"max={stats['max']:.1f}, "
                    f"mean={stats['mean']:.1f}, "
                    f"median={stats['median']:.1f}, "
                    f">20={stats['above_20']}, "
                    f">30={stats['above_30']}, "
                    f">50={stats['above_50']}",
                    throttle_duration_sec=5.0
                )
    
    def extract_high_reflectivity_landmarks(self, scan_msg):
        """
        Extract and cluster high reflectivity points from LaserScan.
        Intensities are 0-255, but typically <50 in your environment.
        
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
        

        # Filter valid points (range > 0 and finite intensity)
        valid_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        valid_mask &= np.isfinite(intensities)
        
        # In ROS LaserScan coordinates:
        # 0 rad → forward
        # +π/2 → left
        # −π/2 → right
        # ±π → directly behind
        # Angle-based FOV filtering (±70 degrees), swap the negtive sign after rotating lidar
        rear_fov_mask = (
        (angles <= np.pi - self.half_fov_rad) |
        (angles >= -np.pi + self.half_fov_rad))
        #valid_mask &= rear_fov_mask

        valid_ranges = ranges[valid_mask]
        valid_intensities = intensities[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return []
        
        # Calculate dynamic threshold based on YOUR typical intensity range (<50)
        max_intensity = np.max(valid_intensities)
        min_intensity = np.min(valid_intensities)
        
        # CRITICAL: Two-part threshold for your data:
        # 1. Absolute threshold (e.g., >20 in 0-255 range)
        # 2. Relative threshold (e.g., >2x median intensity)
        median_intensity = np.median(valid_intensities)
        
        # Dynamic threshold calculation
        absolute_threshold = self.reflectivity_threshold
        relative_threshold = median_intensity * self.relative_threshold_multiplier
        
        # Use whichever is higher to be conservative
        threshold = max(absolute_threshold, relative_threshold)
        
        self.get_logger().debug(
            f"Intensity thresholds: max={max_intensity:.1f}, "
            f"median={median_intensity:.1f}, "
            f"abs_thresh={absolute_threshold}, "
            f"rel_thresh={relative_threshold:.1f}, "
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
        z = np.full_like(x, self.sensor_height)  # Creates array of same shape as x, filled with sensor_height
        
        # Adaptive clustering: use larger radius for distant points
        # (points further away are more sparse in Cartesian space)
        landmarks = []
        processed = np.zeros(len(x), dtype=bool)
        
        for i in range(len(x)):
            if processed[i]:
                continue
            
            # Adaptive cluster radius based on distance
            distance = np.sqrt(x[i]**2 + y[i]**2)
            cluster_radius = 0.1 + 0.05 * (distance / 5.0)  # 0.1m at 0m, increases with distance
            
            # Find points close to this one
            distances = np.sqrt((x - x[i])**2 + (y - y[i])**2)
            cluster_mask = distances < cluster_radius
            
            # Create landmark from cluster if large enough
            cluster_size = np.sum(cluster_mask)
            if cluster_size >= self.min_cluster_size:
                cluster_x = np.mean(x[cluster_mask])
                cluster_y = np.mean(y[cluster_mask])
                cluster_z = np.mean(z[cluster_mask])
                cluster_intensity = np.mean(high_intensities[cluster_mask])
                
                # Normalize intensity to 0-1 for consistent processing
                # Since max is typically <50, normalize to 0-100 scale
                intensity_normalized = min(cluster_intensity / 100.0, 1.0)
                
                landmarks.append({
                    'x': float(cluster_x),
                    'y': float(cluster_y),
                    'z': float(cluster_z),
                    'intensity': float(cluster_intensity),  # Original 0-255
                    'intensity_normalized': float(intensity_normalized),  # Normalized 0-1
                    'cluster_size': int(cluster_size),
                    'distance': float(distance),
                    'raw_points': list(zip(x[cluster_mask], y[cluster_mask]))
                })
                
                processed[cluster_mask] = True
                self.stats['landmarks_created'] += 1
        
        return landmarks
    
    def image_callback(self, msg):
        """Augment image with markers at reflectivity landmark locations."""
        if self.K is None or not self.current_landmarks:
            # Pass through if no calibration or landmarks
            self.pub_augmented.publish(msg)
            return
        
        try:
            # Convert to OpenCV
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            augmented = self.cv_image.copy()
            debug = self.cv_image.copy()
            
            # Get transform from lidar to camera
            # try:
            #     transform = self.tf_buffer.lookup_transform(
            #         self.lidar_frame, self.camera_frame,
            #         rclpy.time.Time())
            #     self.get_logger().debug(f"transform: {transform}")
            
            # except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            #         tf2_ros.ExtrapolationException) as e:
            #     self.get_logger().warn(f"TF lookup failed: {e}")
            #     self.pub_augmented.publish(msg)
            #     return
            
            # Process each landmark
            markers_added = 0
            valid_landmarks = []
            
            for landmark in self.current_landmarks:
                # Project landmark to image
                uv = self.project_to_image(landmark)
                if uv is None:
                    continue
                
                u, v = uv
                
                # Check if within image bounds
                if 0 <= u < self.cv_image.shape[1] and 0 <= v < self.cv_image.shape[0]:
                    valid_landmarks.append({
                        'uv': (u, v),
                        'landmark': landmark,
                        'marker_id': self.get_marker_id(landmark)
                    })
            
            if not valid_landmarks:
                self.get_logger().debug("No landmarks projected to image")
                self.pub_augmented.publish(msg)
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
                # Use normalized intensity for color mapping
                intensity_norm = landmark['intensity_normalized']
                color_intensity = int(intensity_norm * 255)
                
                # Color gradient: blue (low) -> green (medium) -> red (high)
                if intensity_norm < 0.33:
                    color = (255, int(color_intensity * 3), 0)  # Blue to cyan
                elif intensity_norm < 0.66:
                    color = (255 - int(color_intensity * 1.5), 255, 0)  # Cyan to green
                else:
                    color = (0, 255 - int(color_intensity * 0.5), color_intensity)  # Green to red
                
                cv2.circle(debug, (u, v), 8, color, 2)
                cv2.putText(debug, f"{landmark['intensity']:.0f}", 
                           (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
                
                markers_added += 1
            
            # Publish augmented image
            augmented_msg = self.bridge.cv2_to_imgmsg(augmented, encoding='bgr8')
            augmented_msg.header = msg.header
            self.pub_augmented.publish(augmented_msg)
            
            # Publish debug visualization with statistics
            debug_stats = (
                f"Markers: {markers_added}/{len(valid_landmarks)} | "
                f"Landmarks: {len(self.current_landmarks)}"
            )
            cv2.putText(debug, debug_stats, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_msg.header = msg.header
            self.pub_debug.publish(debug_msg)
            
            self.get_logger().debug(
                f"Added {markers_added} synthetic markers from {len(self.current_landmarks)} landmarks",
                throttle_duration_sec=1.0
            )
            
        except Exception as e:
            self.get_logger().error(f"Augmentation failed: {e}")
            # Pass through original image on error
            self.pub_augmented.publish(msg)
        
    def project_to_image(self, landmark):
        """
        Project a world-anchored LiDAR landmark into camera image pixels.

        Pipeline:
        LiDAR → MAP → CAMERA_OPTICAL → IMAGE
        """

        if self.K is None:
            self.get_logger().warn("Camera intrinsics not available")
            return None

        try:
            #get lidar to map transform
            transform_l_m = self.tf_buffer.lookup_transform(
            self.map_frame,  # target
            self.lidar_frame,   # source
            rclpy.time.Time()
        )
            #get map to camera transform
            transform_m_c = self.tf_buffer.lookup_transform(
            self.camera_frame,  # target
            self.map_frame,   # source
            rclpy.time.Time()
        )
                 
            p_lidar = PointStamped()
            p_lidar.header.frame_id = self.lidar_frame
            #p_lidar.header.stamp= msg.header.stamp
            p_lidar.point.x = landmark["x"]
            p_lidar.point.y = landmark["y"]
            p_lidar.point.z = landmark["z"]



            p_map = do_transform_point(p_lidar, transform_l_m)
            self.get_logger().info(f"p_map: {p_map}")
            
            p_cam = do_transform_point(p_map, transform_m_c)
            self.get_logger().info(f"p_cam: {p_cam}")


            X = p_cam.point.x
            Y = p_cam.point.y
            Z = p_cam.point.z


                #  Alternative method using transform from Lidar to cam directly
             #  better to move thr transform_cam block to the top of script for efficency since its static
            # transform_cam = self.tf_buffer.lookup_transform(
            #     self.cam_frame,  # target
            #     self.lidar_frame,   # source
            #     rclpy.time.Time()
            # )
            #     t = transform_cam.transform.translation
            #     q = transform_cam.transform.rotation

                
            #     translation = np.array([t.x, t.y, t.z])
            #     quaternion_xyzw = np.array([q.x, q.y, q.z, q.w])
                
            #     self.get_logger().info(f"Translation from TF: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
            #     self.get_logger().info(f"Quaternion from TF: [{quaternion_xyzw[0]:.6f}, {quaternion_xyzw[1]:.6f}, {quaternion_xyzw[2]:.6f}, {quaternion_xyzw[3]:.6f}]")
                
            #    # Create rotation matrix
            #     rotation = Rotation.from_quat(quaternion_xyzw)
            #     R = rotation.as_matrix()
            #     self.get_logger().warn(f"TF rot : {R}")
            
            #             # Build 4x4 transformation matrix
            #     T = np.eye(4)
            #     T[:3, :3] = R
            #     T[:3, 3] = translation


                    # Build point in LiDAR frame
            # point_lidar = np.array([
            #     landmark['x'],
            #     landmark['y'],
            #     landmark['z']
            # ])

            # point_lidar_h = np.append(point_lidar, 1.0)
            # point_cam_h = T @ point_lidar_h
            #X, Y, Z = point_cam_h[:3]

            

            # Camera optical frame: Z forward
            self.get_logger().info(f"Z: {Z}")

            if Z <= 0.01:
                return None

            # 3) Camera projection
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]

            u = int(fx * (X / Z) + cx)
            v = int(fy * (Y / Z) + cy)
            self.get_logger().info(f" u , v: ({u}, {v})")

            image_width = self.image_width
            image_height = self.image_height


            # 4) Image bounds check
            if 0 <= u < self.image_width and 0 <= v < self.image_height:
                self.get_logger().info(f"✅ VALID: Within image ({image_width}x{image_height})")
                return (u, v)

            else:
                # Provide detailed feedback
                out_msg = f"❌ OUTSIDE {image_width}x{image_height}: "
                if u < 0:
                    out_msg += f"u={u} ({abs(u)}px left), "
                elif u >= image_width:
                    out_msg += f"u={u} ({u-image_width+1}px right), "
                if v < 0:
                    out_msg += f"v={v} ({abs(v)}px above), "
                elif v >= image_height:
                    out_msg += f"v={v} ({v-image_height+1}px below), "
                
                out_msg = out_msg.rstrip(", ")
                self.get_logger().warn(out_msg)
                return None


        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ) as e:
            self.get_logger().warn(f"TF error during projection: {e}")
            return None  


        
    def get_marker_id(self, landmark):
        """Generate consistent marker ID for a landmark."""
        # Use quantized position and intensity as ID
        grid_size = 0.1  # 10cm grid
        
        # Quantize position
        x_idx = int(landmark['x'] / grid_size)
        y_idx = int(landmark['y'] / grid_size)
        
        # Quantize intensity (0-10 scale based on normalized intensity)
        # Since intensities are typically <50, map to 0-10 scale appropriately
        intensity_norm = landmark['intensity_normalized']
        intensity_idx = min(9, int(intensity_norm * 10))
        
        # Include cluster size for uniqueness
        cluster_idx = min(9, landmark['cluster_size'])
        
        # Include distance for uniqueness
        distance_idx = min(9, int(landmark['distance']))
        
        return f"{x_idx}_{y_idx}_{intensity_idx}_{cluster_idx}_{distance_idx}"
    
    def get_marker_pattern(self, marker_id):
        """Get or create marker pattern for a given ID."""
        if marker_id not in self.marker_db:
            # Generate new marker pattern
            pattern = self.generate_marker_pattern(marker_id)
            self.marker_db[marker_id] = pattern
            
            # Keep DB size manageable
            if len(self.marker_db) > 100:
                # Remove oldest entry
                oldest_key = next(iter(self.marker_db))
                del self.marker_db[oldest_key]
        
        return self.marker_db[marker_id]
    
    def generate_marker_pattern(self, marker_id):
        """
        Generate a distinctive marker pattern optimized for ORB detection.
        
        ORB loves:
        - High contrast corners
        - Asymmetric patterns
        - Multiple scales
        - Binary intensity transitions
        """
        size = self.marker_size
        pattern = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Use marker_id to seed deterministic but varied patterns
        seed = abs(hash(marker_id)) % 10000
        np.random.seed(seed)
        
        # Choose pattern type - bias towards checkerboard for more corners
        pattern_types = ['checkerboard', 'circles', 'binary', 'cross']
        pattern_type = np.random.choice(pattern_types)
        
        if pattern_type == 'checkerboard':
            # Checkerboard (excellent for ORB corners)
            # Vary cell size for scale invariance
            cell_size_options = [4, 5, 6, 8, 10]
            cell_size = np.random.choice(cell_size_options)
            
            for i in range(0, size, cell_size):
                for j in range(0, size, cell_size):
                    if ((i//cell_size) + (j//cell_size)) % 2 == 0:
                        color = (255, 255, 255)  # White
                    else:
                        color = (0, 0, 0)  # Black
                    pattern[i:min(i+cell_size, size), 
                           j:min(j+cell_size, size)] = color
        
        elif pattern_type == 'circles':
            # Concentric circles with spokes
            pattern.fill(255)  # White background
            center = size // 2
            
            # Draw alternating circles (creates edges)
            num_circles = np.random.randint(3, 6)
            for r in np.linspace(5, size//2 - 5, num_circles):
                color = 0 if (int(r) // 5) % 2 == 0 else 255
                thickness = np.random.choice([1, 2])
                cv2.circle(pattern, (center, center), int(r), 
                          (color, color, color), thickness)
            
            # Add radial lines (creates corners!)
            num_lines = np.random.randint(4, 12)
            line_thickness = np.random.choice([1, 2])
            for angle in np.linspace(0, 2*np.pi, num_lines, endpoint=False):
                length = size//2 - 5
                x2 = center + int(length * np.cos(angle))
                y2 = center + int(length * np.sin(angle))
                cv2.line(pattern, (center, center), (x2, y2), 
                        (0, 0, 0), line_thickness)
        
        elif pattern_type == 'binary':
            # Binary code pattern (unique per marker)
            binary_hash = hash(marker_id)
            binary_str = format(abs(binary_hash) & 0xFFFF, '016b')
            
            # Create 4x4 grid from binary string
            grid_size = size // 4
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    if idx < len(binary_str) and binary_str[idx] == '1':
                        color = (255, 255, 255)
                    else:
                        color = (0, 0, 0)
                    
                    y1, y2 = i*grid_size, (i+1)*grid_size
                    x1, x2 = j*grid_size, (j+1)*grid_size
                    pattern[y1:y2, x1:x2] = color
        
        else:  # 'cross'
            # Cross pattern with enhancements
            pattern.fill(255)
            center = size // 2
            
            # Draw cross with varying thickness
            cross_thickness = np.random.choice([2, 3])
            cross_length = np.random.randint(size//3, size//2)
            
            cv2.line(pattern, (center-cross_length, center), 
                    (center+cross_length, center), 
                    (0, 0, 0), cross_thickness)
            cv2.line(pattern, (center, center-cross_length), 
                    (center, center+cross_length), 
                    (0, 0, 0), cross_thickness)
            
            # Add corner dots for more features
            dot_radius = np.random.choice([2, 3])
            offset = cross_length - 5
            positions = [
                (center-offset, center-offset),
                (center+offset, center-offset),
                (center-offset, center+offset),
                (center+offset, center+offset),
            ]
            for pos in positions:
                cv2.circle(pattern, pos, dot_radius, (0, 0, 0), -1)
        
        # Add subtle noise (helps with scale invariance)
        if np.random.rand() < 0.4:
            noise_intensity = np.random.randint(10, 25)
            noise = np.random.randint(-noise_intensity, noise_intensity+1, 
                                     (size, size, 3), dtype=np.int16)
            pattern = np.clip(pattern.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Ensure good contrast for ORB
        gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        if contrast < 40:  # Low contrast, enhance
            # Histogram equalization on grayscale
            gray_eq = cv2.equalizeHist(gray)
            pattern = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        
        return pattern
    
    def blend_marker(self, image, center_u, center_v, marker):
        """Blend marker pattern onto image at specified location."""
        h, w = marker.shape[:2]
        half_h, half_w = h // 2, w // 2
        
        # Calculate ROI bounds
        y1 = max(0, center_v - half_h)
        y2 = min(image.shape[0], center_v + half_h)
        x1 = max(0, center_u - half_w)
        x2 = min(image.shape[1], center_u + half_w)
        
        # Calculate corresponding marker region
        m_y1 = max(0, half_h - (center_v - y1))
        m_y2 = min(h, half_h + (y2 - center_v))
        m_x1 = max(0, half_w - (center_u - x1))
        m_x2 = min(w, half_w + (x2 - center_u))
        
        # Extract regions
        roi = image[y1:y2, x1:x2]
        marker_region = marker[m_y1:m_y2, m_x1:m_x2]
        
        # Ensure same size
        if marker_region.shape[:2] != roi.shape[:2]:
            marker_region = cv2.resize(marker_region, 
                                      (roi.shape[1], roi.shape[0]))
        
        # Alpha blending with edge feathering
        alpha = self.marker_opacity
        
        # Optional: create soft mask for smoother blending
        if roi.shape[0] > 10 and roi.shape[1] > 10:
            # Create Gaussian mask for feathering
            mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.float32)
            border = 3
            mask[:border, :] = 0.3
            mask[-border:, :] = 0.3
            mask[:, :border] = 0.3
            mask[:, -border:] = 0.3
            
            # Expand to 3 channels
            mask_3d = np.stack([mask, mask, mask], axis=2)
            alpha_adjusted = alpha * mask_3d
        else:
            alpha_adjusted = alpha
        
        blended = roi * (1 - alpha_adjusted) + marker_region * alpha_adjusted
        blended = blended.astype(np.uint8)
        
        # Copy back to image
        image[y1:y2, x1:x2] = blended

def main(args=None):
    rclpy.init(args=args)
    node = VisualAugmentor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
        # Print final statistics
        node.get_logger().info(f"Final stats: Scans={node.stats['scans_processed']}, "
                              f"Points={node.stats['total_points']}, "
                              f"HighReflect={node.stats['high_reflectivity_points']}, "
                              f"Landmarks={node.stats['landmarks_created']}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()




# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, LaserScan, CameraInfo
# from geometry_msgs.msg import PointStamped
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import tf2_ros
# from tf2_geometry_msgs import do_transform_point

# class VisualAugmentor(Node):
#     """
#     Augments camera images with synthetic markers at reflectivity landmark locations.
#     Processes LaserScan directly to extract high-reflectivity landmarks.
#     """
    
#     def __init__(self):
#         super().__init__('visual_augmentor')
        
#         # Enable ALL debug logging
#         self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)


#         # Subscribers
#         self.sub_image = self.create_subscription(
#             Image, '/camera_optical/image_raw', self.image_callback, 10)
        
#         self.sub_scan = self.create_subscription(
#             LaserScan, '/lidar/scan', self.scan_callback, 10)
        
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
        
#         # Camera parameters
#         self.K = None
#         self.D = None
#         self.camera_frame = None
        
#         # Lidar parameters
#         self.lidar_frame = 'lidar'  # Adjust based on your TF tree
        
#         # Current state
#         self.current_landmarks = []  # List of dictionaries with x, y, z, intensity
#         self.marker_db = {}  # Stores marker patterns for consistency
        
#         # Parameters
#         self.marker_size = 40  # pixels
#         self.marker_opacity = 0.6  # Blend with original image
#         self.reflectivity_threshold = 0.7  # Relative to max intensity
#         self.min_intensity_absolute = 0.3  # Minimum absolute intensity
        
#         # Thread safety
#         self.landmarks_lock = False
        
#         self.get_logger().info("Visual Augmentor Initialized")
    
#     def camera_info_callback(self, msg):
#         """Store camera calibration parameters."""
#         self.K = np.array(msg.k).reshape(3, 3)
#         self.D = np.array(msg.d)
#         self.camera_frame = msg.header.frame_id
#         self.get_logger().info("Camera calibration received")
    
#     def scan_callback(self, msg):
#         """Process laser scan to extract reflectivity landmarks."""
#         try:
#             landmarks = self.extract_high_reflectivity_landmarks(msg)
            
#             # Update landmarks (with simple lock to prevent race condition)
#             self.current_landmarks = landmarks
            
#             self.get_logger().debug(
#                 f"Extracted {len(landmarks)} reflectivity landmarks",
#                 throttle_duration_sec=2.0
#             )
            
#         except Exception as e:
#             self.get_logger().error(f"Scan processing failed: {e}")
    
#     def extract_high_reflectivity_landmarks(self, scan_msg):
#         """
#         Extract and cluster high reflectivity points from LaserScan.
        
#         Returns: List of dictionaries with keys:
#             'x', 'y', 'z', 'intensity', 'cluster_size'
#         """
#         ranges = np.array(scan_msg.ranges)
#         intensities = np.array(scan_msg.intensities)
        
#         if len(intensities) == 0:
#             return []
        
#         # Create angle array
#         angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
#         # Filter valid points
#         valid_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
#         valid_ranges = ranges[valid_mask]
#         valid_intensities = intensities[valid_mask]
#         valid_angles = angles[valid_mask]
        
#         if len(valid_ranges) == 0:
#             return []
        
#         # Calculate intensity threshold
#         max_intensity = np.max(valid_intensities)
#         if max_intensity > 0:
#             threshold = max(self.reflectivity_threshold * max_intensity, 
#                           self.min_intensity_absolute)
#         else:
#             threshold = self.min_intensity_absolute
        
#         # Find high reflectivity points
#         high_reflectivity_mask = valid_intensities > threshold
#         high_ranges = valid_ranges[high_reflectivity_mask]
#         high_intensities = valid_intensities[high_reflectivity_mask]
#         high_angles = valid_angles[high_reflectivity_mask]
        
#         if len(high_ranges) == 0:
#             return []
        
#         # Convert to Cartesian coordinates
#         x = high_ranges * np.cos(high_angles)
#         y = high_ranges * np.sin(high_angles)
#         z = np.zeros_like(x)  # 2D lidar
        
#         # Simple clustering: group points within 0.2m of each other
#         landmarks = []
#         processed = np.zeros(len(x), dtype=bool)
        
#         for i in range(len(x)):
#             if processed[i]:
#                 continue
            
#             # Find points close to this one
#             distances = np.sqrt((x - x[i])**2 + (y - y[i])**2)
#             cluster_mask = distances < 0.2
            
#             # Create landmark from cluster
#             if np.sum(cluster_mask) >= 2:  # Minimum cluster size
#                 cluster_x = np.mean(x[cluster_mask])
#                 cluster_y = np.mean(y[cluster_mask])
#                 cluster_z = np.mean(z[cluster_mask])
#                 cluster_intensity = np.mean(high_intensities[cluster_mask])
#                 cluster_size = np.sum(cluster_mask)
                
#                 landmarks.append({
#                     'x': float(cluster_x),
#                     'y': float(cluster_y),
#                     'z': float(cluster_z),
#                     'intensity': float(cluster_intensity),
#                     'cluster_size': int(cluster_size),
#                     'raw_points': list(zip(x[cluster_mask], y[cluster_mask]))
#                 })
                
#                 processed[cluster_mask] = True
        
#         return landmarks
    
#     def image_callback(self, msg):
#         """Augment image with markers at reflectivity landmark locations."""
#         if self.K is None or not self.current_landmarks:
#             # Pass through if no calibration or landmarks
#             self.pub_augmented.publish(msg)
#             return
        
#         try:
#             # Convert to OpenCV
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             augmented = cv_image.copy()
#             debug = cv_image.copy()
            
#             # Get transform from lidar to camera
#             try:
#                 transform = self.tf_buffer.lookup_transform(
#                     self.camera_frame, self.lidar_frame, 
#                     rclpy.time.Time())
#             except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
#                     tf2_ros.ExtrapolationException) as e:
#                 self.get_logger().warn(f"TF lookup failed: {e}")
#                 self.pub_augmented.publish(msg)
#                 return
            
#             # Process each landmark
#             markers_added = 0
#             valid_landmarks = []
            
#             for landmark in self.current_landmarks:
#                 # Project landmark to image
#                 uv = self.project_to_image(landmark, transform)
#                 if uv is None:
#                     continue
                
#                 u, v = uv
                
#                 # Check if within image bounds
#                 if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
#                     valid_landmarks.append({
#                         'uv': (u, v),
#                         'landmark': landmark,
#                         'marker_id': self.get_marker_id(landmark)
#                     })
            
#             # Sort by intensity (strongest first)
#             valid_landmarks.sort(key=lambda x: x['landmark']['intensity'], reverse=True)
            
#             # Limit number of markers to avoid clutter
#             max_markers = min(10, len(valid_landmarks))
            
#             for i in range(max_markers):
#                 data = valid_landmarks[i]
#                 u, v = data['uv']
#                 landmark = data['landmark']
                
#                 # Create or retrieve marker
#                 marker_pattern = self.get_marker_pattern(data['marker_id'])
                
#                 # Blend marker onto image
#                 self.blend_marker(augmented, u, v, marker_pattern)
                
#                 # Draw debug visualization
#                 color_intensity = int(landmark['intensity'] * 255)
#                 color = (0, color_intensity, 255 - color_intensity)
                
#                 cv2.circle(debug, (u, v), 8, color, 2)
#                 cv2.putText(debug, f"{landmark['intensity']:.2f}", 
#                            (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 
#                            0.5, color, 1)
                
#                 markers_added += 1
            
#             # Publish augmented image
#             augmented_msg = self.bridge.cv2_to_imgmsg(augmented, encoding='bgr8')
#             augmented_msg.header = msg.header
#             self.pub_augmented.publish(augmented_msg)
            
#             # Publish debug visualization
#             cv2.putText(debug, f"Markers: {markers_added}", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
#                        1.0, (0, 255, 0), 2)
#             debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
#             debug_msg.header = msg.header
#             self.pub_debug.publish(debug_msg)
            
#             self.get_logger().debug(
#                 f"Added {markers_added} synthetic markers from {len(self.current_landmarks)} landmarks",
#                 throttle_duration_sec=1.0
#             )
            
#         except Exception as e:
#             self.get_logger().error(f"Augmentation failed: {e}")
#             # Pass through original image on error
#             self.pub_augmented.publish(msg)
    
#     def project_to_image(self, landmark, transform):
#         """Project 3D landmark to 2D image coordinates."""
#         try:
#             # Create 3D point in lidar frame
#             pt_lidar = PointStamped()
#             pt_lidar.point.x = landmark['x']
#             pt_lidar.point.y = landmark['y']
#             pt_lidar.point.z = landmark['z']
#             pt_lidar.header.frame_id = self.lidar_frame
#             pt_lidar.header.stamp = self.get_clock().now().to_msg()
            
#             # Transform to camera frame
#             pt_camera = do_transform_point(pt_lidar, transform)
            
#             # Project to 2D
#             point_3d = np.array([[pt_camera.point.x, pt_camera.point.y, pt_camera.point.z]])
#             uv, _ = cv2.projectPoints(point_3d, 
#                                      np.zeros(3),  # rotation vector
#                                      np.zeros(3),  # translation vector
#                                      self.K, self.D)
            
#             u, v = uv[0][0].astype(int)
#             return (u, v)
            
#         except Exception as e:
#             self.get_logger().debug(f"Projection failed: {e}")
#             return None
    
#     def get_marker_id(self, landmark):
#         """Generate consistent marker ID for a landmark."""
#         # Use quantized position and intensity as ID
#         grid_size = 0.1  # 10cm grid
        
#         # Quantize position
#         x_idx = int(landmark['x'] / grid_size)
#         y_idx = int(landmark['y'] / grid_size)
        
#         # Quantize intensity (0-10 scale)
#         intensity_idx = min(9, int(landmark['intensity'] * 10))
        
#         # Include cluster size for uniqueness
#         cluster_idx = min(9, landmark['cluster_size'])
        
#         return f"{x_idx}_{y_idx}_{intensity_idx}_{cluster_idx}"
    
#     def get_marker_pattern(self, marker_id):
#         """Get or create marker pattern for a given ID."""
#         if marker_id not in self.marker_db:
#             # Generate new marker pattern
#             pattern = self.generate_marker_pattern(marker_id)
#             self.marker_db[marker_id] = pattern
            
#             # Keep DB size manageable
#             if len(self.marker_db) > 100:
#                 # Remove least recently accessed
#                 keys = list(self.marker_db.keys())
#                 if keys:
#                     del self.marker_db[keys[0]]
        
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
#         seed = abs(hash(marker_id)) % 10000
#         np.random.seed(seed)
        
#         pattern_type = np.random.choice(['checkerboard', 'circles', 'binary', 'cross'])
        
#         if pattern_type == 'checkerboard':
#             # Checkerboard (excellent for ORB corners)
#             cell_size = max(4, size // np.random.randint(3, 7))
#             for i in range(0, size, cell_size):
#                 for j in range(0, size, cell_size):
#                     if ((i//cell_size) + (j//cell_size)) % 2 == 0:
#                         color = (255, 255, 255)  # White
#                     else:
#                         color = (0, 0, 0)  # Black
#                     pattern[i:min(i+cell_size, size), 
#                            j:min(j+cell_size, size)] = color
        
#         elif pattern_type == 'circles':
#             # Concentric circles with spokes
#             pattern.fill(255)  # White background
#             center = size // 2
            
#             # Draw alternating circles
#             for r in range(3, size//2 - 2, 3):
#                 color = 0 if (r // 3) % 2 == 0 else 255
#                 cv2.circle(pattern, (center, center), r, 
#                           (color, color, color), 1)
            
#             # Add radial lines (creates corners!)
#             num_lines = np.random.randint(4, 9)
#             for angle in np.linspace(0, 2*np.pi, num_lines, endpoint=False):
#                 length = size//2 - 3
#                 x2 = center + int(length * np.cos(angle))
#                 y2 = center + int(length * np.sin(angle))
#                 cv2.line(pattern, (center, center), (x2, y2), 
#                         (0, 0, 0), 1)
        
#         elif pattern_type == 'binary':
#             # Binary code pattern (unique per marker)
#             binary_hash = hash(marker_id)
#             binary_str = format(abs(binary_hash) & 0xFFFF, '016b')
            
#             # Create 4x4 grid from binary string
#             grid_size = size // 4
#             for i in range(4):
#                 for j in range(4):
#                     idx = i * 4 + j
#                     if idx < len(binary_str) and binary_str[idx] == '1':
#                         color = (255, 255, 255)
#                     else:
#                         color = (0, 0, 0)
                    
#                     y1, y2 = i*grid_size, (i+1)*grid_size
#                     x1, x2 = j*grid_size, (j+1)*grid_size
#                     pattern[y1:y2, x1:x2] = color
        
#         else:  # 'cross'
#             # Cross pattern
#             pattern.fill(255)
#             center = size // 2
            
#             # Draw cross
#             cv2.line(pattern, (center-5, center), (center+5, center), 
#                     (0, 0, 0), 2)
#             cv2.line(pattern, (center, center-5), (center, center+5), 
#                     (0, 0, 0), 2)
            
#             # Add corners
#             cv2.circle(pattern, (center-5, center-5), 2, (0, 0, 0), -1)
#             cv2.circle(pattern, (center+5, center-5), 2, (0, 0, 0), -1)
#             cv2.circle(pattern, (center-5, center+5), 2, (0, 0, 0), -1)
#             cv2.circle(pattern, (center+5, center+5), 2, (0, 0, 0), -1)
        
#         # Add subtle noise (helps with scale invariance)
#         if np.random.rand() < 0.3:
#             noise = np.random.randint(-15, 16, (size, size, 3), dtype=np.int16)
#             pattern = np.clip(pattern.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
#         # Ensure contrast
#         if np.std(pattern) < 30:
#             # Boost contrast
#             pattern = cv2.convertScaleAbs(pattern, alpha=1.5, beta=0)
        
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
        
#         # Calculate corresponding marker region
#         m_y1 = max(0, half_h - (center_v - y1))
#         m_y2 = min(h, half_h + (y2 - center_v))
#         m_x1 = max(0, half_w - (center_u - x1))
#         m_x2 = min(w, half_w + (x2 - center_u))
        
#         # Extract regions
#         roi = image[y1:y2, x1:x2]
#         marker_region = marker[m_y1:m_y2, m_x1:m_x2]
        
#         # Ensure same size
#         if marker_region.shape[:2] != roi.shape[:2]:
#             marker_region = cv2.resize(marker_region, 
#                                       (roi.shape[1], roi.shape[0]))
        
#         # Alpha blending
#         alpha = self.marker_opacity
#         blended = cv2.addWeighted(roi, 1-alpha, marker_region, alpha, 0)
        
#         # Copy back to image
#         image[y1:y2, x1:x2] = blended

# def main(args=None):
#     rclpy.init(args=args)
#     node = VisualAugmentor()
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()






