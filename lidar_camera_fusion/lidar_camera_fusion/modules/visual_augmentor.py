#cleaned and refactored to use a single pattern

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
import os
from datetime import datetime
from message_filters import ApproximateTimeSynchronizer, Subscriber




class VisualAugmentor(Node):
    """
    Augments camera images with synthetic markers at reflectivity landmark locations.
    Uses polygon patterns based on normalized intensity values.
    """
    
    def __init__(self):
        super().__init__('visual_augmentor')

        # Enable debug logging
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
        self.pub_augmented = self.create_publisher(Image, '/camera/image_augmented', 10)
        self.pub_debug = self.create_publisher(Image, '/augmentation/debug', 10)
        
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
        
        # Lidar parameters
        self.lidar_frame = 'lidar'
        self.sensor_height = 0
        self.map_frame = 'map'
        
        # Lidar field of view
        fov_deg = 140.0
        self.half_fov_rad = np.deg2rad(fov_deg / 2.0)
        
        # Current state
        self.latest_image = None
        self.latest_image_header = None
        
        # Pattern configuration
        self.marker_size = 40
        self.marker_opacity = 0.9
        self.marker_pattern = self.create_marker_pattern()
        
        # Cache for pre-generated patterns
        # self.pattern_cache = {}
        # self.pre_generate_patterns()
        
        # Intensity parameters
        self.reflectivity_threshold = 0.5
        self.relative_threshold_multiplier = 1.1
        self.min_cluster_size = 3
        
        # Landmark lifecycle management
        self.landmarks = {}
        self.current_scan_landmarks = []
        self.required_observations = 10
        self.spatial_consistency_threshold = 0.3
        self.max_landmark_age = 1.0
        
        # Statistics
        self.stats = {
            'scans_processed': 0,
            'total_points': 0,
            'high_reflectivity_points': 0,
            'landmarks_created': 0,
            'landmarks_confirmed': 0,
            'landmarks_expired': 0,
            'sync_calls': 0,
            'sync_skipped_no_calib': 0,
            'sync_skipped_no_landmarks': 0,
            'sync_skipped_no_confirmed': 0
        }
        
        # FOV intensity tracking
        self.fov_observed_min_intensity = float('inf')
        self.fov_observed_max_intensity = float('-inf')
        
        # Logging
        self.log_file_path = self.create_log_file()
        self.log_intensity_mapping_header()
        self.mapping_counter = 0
        
        self.get_logger().info("Visual Augmentor Initialized")
        self.get_logger().info(f"Confirmation requires: {self.required_observations} observations")
        self.get_logger().info(f"Intensity mapping log file: {self.log_file_path}")
    
    def create_log_file(self):
        """Create a unique log file with timestamp."""
        log_dir = os.path.expanduser('~/april_tag_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"intensity_mapping_{timestamp}.csv"
        return os.path.join(log_dir, filename)
    
    def log_intensity_mapping_header(self):
        """Write header to the log file."""
        try:
            with open(self.log_file_path, 'w') as f:
                f.write("timestamp,intensity_raw,intensity_normalized,intensity_idx,quarter,x_position,y_position,marker_id,fov_min,fov_max\n")
            self.get_logger().info(f"Created log file: {self.log_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to create log file: {e}")
    
    def log_intensity_mapping(self, landmark, intensity_idx, quarter):
        """Log intensity mapping data to file."""
        try:
            self.mapping_counter += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            with open(self.log_file_path, 'a') as f:
                f.write(f"{timestamp},"
                       f"{landmark['intensity']:.2f},"
                       f"{landmark['intensity_normalized']:.4f},"
                       f"{intensity_idx},"
                       f"{quarter},"
                       f"{landmark.get('x', 0):.3f},"
                       f"{landmark.get('y', 0):.3f},"
                       f"{self.fov_observed_min_intensity:.2f},"
                       f"{self.fov_observed_max_intensity:.2f}\n")
        except Exception as e:
            self.get_logger().error(f"Failed to write to log file: {e}")
    
    def camera_info_callback(self, msg):
        """Store camera calibration parameters."""
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)
        self.camera_frame = "camera_optical"
        self.image_width = msg.width
        self.image_height = msg.height
        self.get_logger().info(f"Camera calibration received: {self.image_width}x{self.image_height}")
        
    def sync_callback(self, image_msg, scan_msg):
        """Synchronized callback for image and laser scan."""
        self.stats['sync_calls'] += 1
        
        # Extract raw landmarks from current scan
        self.current_scan_landmarks = self.extract_landmarks(scan_msg)
        
        # Update database (adds candidates, increments observations, confirms)
        self.update_landmark_database()
        
        # Build confirmed landmarks list directly (replace get_confirmed_landmarks)
        confirmed_landmarks = []
        for landmark_id, landmark in self.landmarks.items():
            if landmark['confirmed']:
                confirmed_landmarks.append({
                    'x': landmark['x_map'],
                    'y': landmark['y_map'],
                    'z': landmark['z_map'],
                    'intensity': landmark['intensity'],
                    'intensity_normalized': landmark['intensity_normalized'],
                    'cluster_size': landmark['cluster_size'],
                    'observation_count': landmark['observation_count'],
                    'landmark_id': landmark_id
                })
        
        if self.stats['sync_calls'] % 10 == 0:
            time_diff = abs(
                (image_msg.header.stamp.sec + image_msg.header.stamp.nanosec*1e-9) -
                (scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec*1e-9)
            )
            self.get_logger().info(
                f"Sync #{self.stats['sync_calls']}: Time diff={time_diff:.3f}s, "
                f"Candidates={len(self.current_scan_landmarks)}, "
                f"Confirmed={len(confirmed_landmarks)}/{len(self.landmarks)}"
            )
        
        # Process the synchronized pair with confirmed landmarks
        self.process_synchronized_data(image_msg, confirmed_landmarks)
    
    def update_landmark_database(self):
        """Update persistent landmark database with new observations."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        for landmark_id in self.landmarks:
            self.landmarks[landmark_id]['seen_in_current_scan'] = False
        
        for candidate in self.current_scan_landmarks:
            matched = False
            
            for landmark_id, existing in self.landmarks.items():
                dx = candidate['x'] - existing['x_map']
                dy = candidate['y'] - existing['y_map']
                dz = candidate['z'] - existing['z_map']
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if distance < self.spatial_consistency_threshold:
                    self.landmarks[landmark_id]['observation_count'] += 1
                    self.landmarks[landmark_id]['last_seen'] = current_time
                    self.landmarks[landmark_id]['seen_in_current_scan'] = True
                    
                    alpha = 0.3
                    self.landmarks[landmark_id]['x_map'] = (1-alpha) * existing['x_map'] + alpha * candidate['x']
                    self.landmarks[landmark_id]['y_map'] = (1-alpha) * existing['y_map'] + alpha * candidate['y']
                    self.landmarks[landmark_id]['z_map'] = (1-alpha) * existing['z_map'] + alpha * candidate['z']
                    
                    self.landmarks[landmark_id]['intensity'] = max(existing['intensity'], candidate['intensity'])
                    self.landmarks[landmark_id]['intensity_normalized'] = max(
                        existing['intensity_normalized'], candidate['intensity_normalized']
                    )
                    
                    if not existing['confirmed'] and self.landmarks[landmark_id]['observation_count'] >= self.required_observations:
                        self.landmarks[landmark_id]['confirmed'] = True
                        self.stats['landmarks_confirmed'] += 1
                        self.get_logger().info(f"Landmark {landmark_id} CONFIRMED")
                    
                    matched = True
                    break
            
            if not matched:
                landmark_id = self.generate_landmark_id(candidate)
                self.landmarks[landmark_id] = {
                    'x_map': candidate['x'],
                    'y_map': candidate['y'],
                    'z_map': candidate['z'],
                    'intensity': candidate['intensity'],
                    'intensity_normalized': candidate['intensity_normalized'],
                    'observation_count': 1,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'confirmed': False,
                    'seen_in_current_scan': True,
                    'cluster_size': candidate['cluster_size']
                }
                self.stats['landmarks_created'] += 1
        
        expired_ids = []
        for landmark_id, landmark in self.landmarks.items():
            if current_time - landmark['last_seen'] > self.max_landmark_age:
                expired_ids.append(landmark_id)
        
        for landmark_id in expired_ids:
            del self.landmarks[landmark_id]
            self.stats['landmarks_expired'] += 1

    
    def generate_landmark_id(self, landmark):
        """Generate unique ID based on quantized position."""
        grid_size = 0.1
        x_idx = int(landmark['x'] / grid_size)
        y_idx = int(landmark['y'] / grid_size)
        z_idx = int(landmark['z'] / grid_size)
        return f"lm_{x_idx}_{y_idx}_{z_idx}"
    
    # def process_synchronized_data(self, image_msg, landmarks):
    #     """
    #     Process time-synchronized image and CONFIRMED landmarks only.
    #     """
    #     if self.K is None:
    #         self.stats['sync_skipped_no_calib'] += 1
    #         self.pub_augmented.publish(image_msg)
    #         return
        
    #     if not landmarks:
    #         self.stats['sync_skipped_no_landmarks'] += 1
    #         self.pub_augmented.publish(image_msg)
    #         return
        
    #     confirmed_count = len([l for l in self.landmarks.values() if l['confirmed']])
    #     if confirmed_count == 0:
    #         self.stats['sync_skipped_no_confirmed'] += 1
    #         self.pub_augmented.publish(image_msg)
    #         return
        
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    #         augmented = cv_image.copy()
    #         debug = cv_image.copy()
            
    #         valid_landmarks = []
    #         for landmark in landmarks:
    #             uv = self.get_projection_points(landmark)
    #             if uv is None:
    #                 continue
                
    #             u, v = uv
    #             if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
    #                 valid_landmarks.append({
    #                     'uv': (u, v),
    #                     'landmark': landmark
    #                 })
            
    #         if not valid_landmarks:
    #             self.pub_augmented.publish(image_msg)
    #             return
            
    #         # Sort by normalized intensity (strongest first)
    #         valid_landmarks.sort(key=lambda x: x['landmark']['intensity_normalized'], reverse=True)
            
    #         # Limit number of markers to avoid clutter
    #         max_markers = min(5, len(valid_landmarks))
            
    #         for i in range(max_markers):
    #             data = valid_landmarks[i]
    #             u, v = data['uv']
    #             landmark = data['landmark']
                
    #             # Use the single pattern directly
    #             self.project_pattern(augmented, u, v, self.marker_pattern)
                
    #             # Draw debug visualization
    #             intensity_norm = landmark['intensity_normalized']
    #             color_intensity = int(intensity_norm * 255)
                
    #             if intensity_norm < 0.33:
    #                 color = (255, int(color_intensity * 3), 0)
    #             elif intensity_norm < 0.66:
    #                 color = (255 - int(color_intensity * 1.5), 255, 0)
    #             else:
    #                 color = (0, 255 - int(color_intensity * 0.5), color_intensity)
                
    #             cv2.circle(debug, (u, v), 8, color, 2)
    #             cv2.circle(debug, (u, v), 10, (255, 255, 255), 1)
                
    #             obs_count = landmark.get('observation_count', self.required_observations)
    #             cv2.putText(debug, f"{landmark['intensity']:.0f}({obs_count})", 
    #                     (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    #         # Publish images
    #         augmented_msg = self.bridge.cv2_to_imgmsg(augmented, encoding='bgr8')
    #         augmented_msg.header = image_msg.header
    #         self.pub_augmented.publish(augmented_msg)
            
    #         debug_stats = f"Sync #{self.stats['sync_calls']}: {max_markers}/{len(valid_landmarks)} confirmed | Total: {len(self.landmarks)} ({confirmed_count} confirmed)"
    #         cv2.putText(debug, debug_stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    #         debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
    #         debug_msg.header = image_msg.header
    #         self.pub_debug.publish(debug_msg)
            
    #     except Exception as e:
    #         self.get_logger().error(f"Augmentation failed: {e}")
    #         self.pub_augmented.publish(image_msg)


    def process_synchronized_data(self, image_msg, landmarks):
        """
        Process time-synchronized image and CONFIRMED landmarks only.
        Blurs regions below ALL LiDAR points from the current scan.
        Maintains last known blur cutoff when no high-reflectivity points exist.
        """
        if self.K is None:
            self.stats['sync_skipped_no_calib'] += 1
            self.get_logger().warn("No camera calibration yet, skipping augmentation")
            self.pub_augmented.publish(image_msg)
            return
        
        # Count confirmed landmarks for stats
        confirmed_count = len([l for l in self.landmarks.values() if l['confirmed']])
        
        # Initialize blur cutoff tracking if not exists
        if not hasattr(self, 'last_blur_cutoff'):
            self.last_blur_cutoff = None
            self.get_logger().info("Initialized last_blur_cutoff tracking")
        
        try:
            # Convert to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            augmented = cv_image.copy()
            debug = cv_image.copy()
            
            # ========== BLUR REGIONS BELOW LIDAR POINTS ==========
            # Try to get cutoff from current scan first
            current_cutoff = None
            total_points_projected = 0
            
            self.get_logger().debug(f"Processing {len(self.current_scan_landmarks)} LiDAR points for blurring")
            
            for candidate in self.current_scan_landmarks:
                # Project each LiDAR point to image
                uv = self.get_projection_points(candidate)
                if uv is None:
                    continue
                
                u, v = uv
                total_points_projected += 1
                
                # Check if within image bounds
                if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
                    # Calculate cutoff line: 20 pixels BELOW this point
                    cutoff_y = v - 10
                    
                    # Only consider if cutoff is within image bounds
                    if cutoff_y < cv_image.shape[0]:
                        # Update current_cutoff to the highest point (smallest Y)
                        if current_cutoff is None or cutoff_y < current_cutoff:
                            current_cutoff = cutoff_y
                            self.get_logger().debug(f"New current_cutoff: {current_cutoff} from point at y={v}")
            
            # Decide which cutoff to use
            if current_cutoff is not None:
                # Use current scan's cutoff
                blur_cutoff = current_cutoff
                self.last_blur_cutoff = blur_cutoff  # Store for future use
                self.get_logger().info(f"Using current scan cutoff: y={blur_cutoff} (from {total_points_projected} projected points)")
            elif self.last_blur_cutoff is not None:
                # No high-reflectivity points in this scan, use last known cutoff
                blur_cutoff = self.last_blur_cutoff
                self.get_logger().info(f"No high-reflectivity points in current scan. Using last known cutoff: y={blur_cutoff}")
            else:
                # No cutoff ever established, skip blurring
                blur_cutoff = None
                self.get_logger().info("No cutoff established yet - skipping blurring")
            
            # Apply blurring if we have a valid cutoff
            # if blur_cutoff is not None and blur_cutoff < cv_image.shape[0]:
            #     self.get_logger().info(f"Applying blurring from y={blur_cutoff} to bottom")
                
            #     # Blur from cutoff to bottom of image
            #     roi = augmented[blur_cutoff:, :]
            #     if roi.size > 0:
            #         # Apply strong Gaussian blur
            #         blurred_roi = cv2.GaussianBlur(roi, (31, 31), 15)
            #         augmented[blur_cutoff:, :] = blurred_roi
                    
            #         # Also blur debug image for visualization
            #         debug_roi = debug[blur_cutoff:, :]
            #         blurred_debug_roi = cv2.GaussianBlur(debug_roi, (31, 31), 15)
            #         debug[blur_cutoff:, :] = blurred_debug_roi
                    
            #         # Draw cutoff line on debug image
            #         cv2.line(debug, (0, blur_cutoff), (cv_image.shape[1], blur_cutoff), 
            #                 (0, 0, 255), 2)
                    
            #         # Add status text
            #         if current_cutoff is not None:
            #             status_text = f"BLURRED BELOW (current scan: y={blur_cutoff})"
            #         else:
            #             status_text = f"BLURRED BELOW (maintained: y={blur_cutoff})"
                    
            #         cv2.putText(debug, status_text, 
            #                 (10, blur_cutoff - 5), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 0.4, (0, 0, 255), 1)
            # else:
            #     self.get_logger().info("No blurring applied - no valid cutoff")
            #     cv2.putText(debug, "NO BLURRING - No cutoff established", 
            #             (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
            #             0.5, (0, 0, 255), 1)
            
            # ========== PROCESS MARKERS (only from confirmed landmarks) ==========
            markers_added = 0
            valid_landmarks = []
            
            for landmark in landmarks:
                # Project landmark to image
                uv = self.get_projection_points(landmark)
                if uv is None:
                    continue
                
                u, v = uv
                
                # Check if within image bounds
      
                if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
                    valid_landmarks.append({
                        'uv': (u, v),
                        'landmark': landmark
                    })


            
            if valid_landmarks:
                # Sort by normalized intensity (strongest first)
                valid_landmarks.sort(key=lambda x: x['landmark']['intensity_normalized'], reverse=True)
                
                # Limit number of markers to avoid clutter
                max_markers = min(10, len(valid_landmarks))
                
                for i in range(max_markers):
                    data = valid_landmarks[i]
                    u, v = data['uv']
                    landmark = data['landmark']
                    
                    # Create or retrieve marker
                  #  marker_pattern = self.get_marker_pattern(data['marker_id'], landmark)
                    
                    # Blend marker onto image
                    self.project_pattern(augmented, u, v, self.marker_pattern)
                    
                    # Draw debug visualization
                    intensity_norm = landmark['intensity_normalized']
                    color_intensity = int(intensity_norm * 255)
                    
                    if intensity_norm < 0.33:
                        color = (255, int(color_intensity * 3), 0)
                    elif intensity_norm < 0.66:
                        color = (255 - int(color_intensity * 1.5), 255, 0)
                    else:
                        color = (0, 255 - int(color_intensity * 0.5), color_intensity)
                    
                    cv2.circle(debug, (u, v), 8, color, 2)
                    cv2.circle(debug, (u, v), 10, (255, 255, 255), 1)
                    
                    obs_count = landmark.get('observation_count', self.required_observations)
                    cv2.putText(debug, f"{landmark['intensity']:.0f}({obs_count})", 
                            (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 1)
                    
                    markers_added += 1
            
            # Add debug statistics to image
            blur_source = "current" if current_cutoff is not None else ("maintained" if self.last_blur_cutoff is not None else "none")
            debug_stats = (
                f"Sync #{self.stats['sync_calls']}: {markers_added} markers | "
                f"Confirmed: {confirmed_count} | "
                f"LiDAR pts: {len(self.current_scan_landmarks)} | "
                f"Blur: y={blur_cutoff if blur_cutoff else 'N/A'} ({blur_source})"
            )
            cv2.putText(debug, debug_stats, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)
            
            # Publish augmented image
            augmented_msg = self.bridge.cv2_to_imgmsg(augmented, encoding='bgr8')
            augmented_msg.header = image_msg.header
            self.pub_augmented.publish(augmented_msg)
            
            # Publish debug visualization
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.pub_debug.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"Augmentation failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.pub_augmented.publish(image_msg)



    def extract_landmarks(self, scan_msg):
        """Extract high reflectivity landmarks from laser scan."""
        ranges = np.array(scan_msg.ranges)
        intensities = np.array(scan_msg.intensities, dtype=np.float32)
        
        self.stats['total_points'] += len(intensities)
        
        if len(intensities) == 0:
            return []
        
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        
        # Basic filters for normalization
        range_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        finite_mask = np.isfinite(intensities)
        valid_mask = range_mask & finite_mask
        
        intensities_all = intensities[valid_mask]
        if len(intensities_all) == 0:
            return []
        
        min_intensity = np.min(intensities_all)
        max_intensity = np.max(intensities_all)
        
        if max_intensity > min_intensity:
            norm_intensities = (intensities_all - min_intensity) / (max_intensity - min_intensity)
        else:
            norm_intensities = np.zeros_like(intensities_all)
        
        # Apply FOV filter
        fov_mask = (angles >= -self.half_fov_rad) & (angles <= self.half_fov_rad)
        final_mask = valid_mask & fov_mask
        
        valid_ranges = ranges[final_mask]
        valid_intensities = intensities[final_mask]
        valid_angles = angles[final_mask]
        
        # Get normalized intensities for FOV points
        valid_indices = np.where(valid_mask)[0]
        fov_indices = valid_indices[fov_mask[valid_mask]]
        
        if len(fov_indices) > 0:
            fov_norm = norm_intensities[fov_mask[valid_mask]]
        else:
            fov_norm = np.array([])
        
        if len(valid_ranges) == 0:
            return []
        
        self.fov_observed_min_intensity = np.min(valid_intensities)
        self.fov_observed_max_intensity = np.max(valid_intensities)
        
        median_intensity = np.median(fov_norm)
        threshold = max(self.reflectivity_threshold, median_intensity * self.relative_threshold_multiplier)
        
        high_mask = fov_norm > threshold
        self.stats['high_reflectivity_points'] += np.sum(high_mask)
        
        high_ranges = valid_ranges[high_mask]
        high_norm = fov_norm[high_mask]
        high_raw = valid_intensities[high_mask]
        high_angles = valid_angles[high_mask]
        
        if len(high_ranges) == 0:
            return []
        
        x = high_ranges * np.cos(high_angles)
        y = high_ranges * np.sin(high_angles)
        z = np.full_like(x, self.sensor_height)
        
        # TF transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.lidar_frame, scan_msg.header.stamp,
                rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, self.lidar_frame, rclpy.time.Time()
                )
            except Exception as e2:
                self.get_logger().error(f"TF fallback failed: {e2}")
                return []
        
        # Create landmarks
        landmarks = []
        processed = np.zeros(len(x), dtype=bool)
        
        for i in range(len(x)):
            if processed[i]:
                continue
            
            distance = np.sqrt(x[i]**2 + y[i]**2)
            cluster_radius = 0.3 + 0.05 * (distance / 5.0)
            
            dists = np.sqrt((x - x[i])**2 + (y - y[i])**2)
            cluster_mask = dists < cluster_radius
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size >= self.min_cluster_size:
                cluster_x = np.max(x[cluster_mask])
                cluster_y = np.max(y[cluster_mask])
                cluster_z = np.max(z[cluster_mask])
                cluster_norm = np.mean(high_norm[cluster_mask])
                cluster_raw = np.mean(high_raw[cluster_mask])
                
                p_lidar = PointStamped()
                p_lidar.header.frame_id = self.lidar_frame
                p_lidar.header.stamp = scan_msg.header.stamp
                p_lidar.point.x = cluster_x
                p_lidar.point.y = cluster_y
                p_lidar.point.z = cluster_z
                
                p_map = do_transform_point(p_lidar, transform)

                 # Log intensity mapping 
                if self.mapping_counter < 100:  # Limit logging to first 100 landmarks
                    intensity_idx = min(9, int(cluster_norm * 10))
                    quarter = 0 if intensity_idx <= 2 else 1 if intensity_idx <= 5 else 2 if intensity_idx <= 8 else 3
                    self.log_intensity_mapping({
                        'intensity': cluster_raw,
                        'intensity_normalized': cluster_norm,
                        'x': p_map.point.x,
                        'y': p_map.point.y
                    }, intensity_idx, quarter)
                
                landmarks.append({
                    'x': p_map.point.x,
                    'y': p_map.point.y,
                    'z': p_map.point.z,
                    'intensity': float(cluster_raw),
                    'intensity_normalized': float(cluster_norm),
                    'cluster_size': 1,
                    'distance': float(distance)
                })
                
                processed[cluster_mask] = True
                self.stats['landmarks_created'] += 1
        
        return landmarks
    
    def get_projection_points(self, landmark):
        """Project landmark to camera image coordinates."""
        if self.K is None:
            return None
        
        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame, self.map_frame, rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            
            p_map = PointStamped()
            p_map.header.frame_id = self.map_frame
            p_map.point.x = landmark["x"]
            p_map.point.y = landmark["y"]
            p_map.point.z = landmark["z"]
            
            p_cam = do_transform_point(p_map, transform)
            
            X, Y, Z = p_cam.point.x, p_cam.point.y, p_cam.point.z
            
            if Z <= 0.01:
                return None
            
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            
            u = int(fx * (X / Z) + cx)
            v = int(fy * (Y / Z) + cy)
            
            if 0 <= u < self.image_width and 0 <= v < self.image_height:
                return (u, v)
            return None
            
        except Exception as e:
            self.get_logger().warn(f"Projection error: {e}")
            return None
    

    def create_marker_pattern(self):
        """
        Generate a simple pattern: square with a black square inside.
        """
        try:
            # Create white square background
            marker = np.ones((self.marker_size, self.marker_size, 3), dtype=np.uint8) * 255
            
            # Calculate center and size
            center = (self.marker_size // 2, self.marker_size // 2)
            polygon_size = self.marker_size // 2  # Polygon takes 1/3 of marker size

            half = polygon_size // 2
            top_left = (center[0] - half, center[1] - half)
            bottom_right = (center[0] + half, center[1] + half)
            cv2.rectangle(marker, top_left, bottom_right, (0, 0, 0), -1)
                
            
            # Add a thin black border around the entire marker for contrast
            cv2.rectangle(marker, (0, 0), (self.marker_size-1, self.marker_size-1), (0, 0, 0), 1)
            
            return marker
            
        except Exception as e:
            self.get_logger().error(f"Failed to generate pattern for tag_id {tag_id}: {e}")
            # Fallback: simple black square
            fallback = np.zeros((self.marker_size, self.marker_size, 3), dtype=np.uint8)
            cv2.rectangle(fallback, (5, 5), (self.marker_size-5, self.marker_size-5), (255, 255, 255), -1)
            return fallback


        # """
        # Create a 4x4 checkerboard pattern (alternating black and white squares).
        # Called once during initialization.
        # """
        # try:
        #     # Create white square background
        #     marker = np.ones((self.marker_size, self.marker_size, 3), dtype=np.uint8) * 255
            
        #     # Number of cells per row/column (4x4)
        #     cells = 4
        #     cell_size = self.marker_size // cells  # 40 // 4 = 10 pixels
            
        #     # Draw checkerboard pattern
        #     for i in range(cells):
        #         for j in range(cells):
        #             # Determine if this cell should be black (alternating pattern)
        #             if (i + j) % 2 == 0:
        #                 # Calculate cell boundaries
        #                 x1 = j * cell_size
        #                 y1 = i * cell_size
        #                 x2 = x1 + cell_size
        #                 y2 = y1 + cell_size
        #                 # Draw black square
        #                 cv2.rectangle(marker, (x1, y1), (x2, y2), (0, 0, 0), -1)
            
        #     # Add thin black border around entire marker for contrast
        #     cv2.rectangle(marker, (0, 0), (self.marker_size-1, self.marker_size-1), (0, 0, 0), 1)
            
        #     return marker
            
        # except Exception as e:
        #     self.get_logger().error(f"Failed to create marker pattern: {e}")
        #     # Fallback: simple black square with white inner square
        #     fallback = np.zeros((self.marker_size, self.marker_size, 3), dtype=np.uint8)
        #     cv2.rectangle(fallback, (5, 5), (self.marker_size-5, self.marker_size-5), (255, 255, 255), -1)
        #     return fallback
    
    # def create_marker_pattern(self):
    #     """
    #     Create a single pattern: white square with black inner square.
    #     Called once during initialization.
    #     """
    #     try:
    #         # Create white square background
    #         marker = np.ones((self.marker_size, self.marker_size, 3), dtype=np.uint8) * 255
            
    #         # Calculate center and size for inner square
    #         center = self.marker_size // 2
    #         square_size = self.marker_size // 2
    #         half = square_size // 2
            
    #         # Draw black inner square
    #         top_left = (center - half, center - half)
    #         bottom_right = (center + half, center + half)
    #         cv2.rectangle(marker, top_left, bottom_right, (0, 0, 0), -1)
            
    #         # Add thin black border around entire marker
    #         cv2.rectangle(marker, (0, 0), (self.marker_size-1, self.marker_size-1), (0, 0, 0), 1)
            
    #         return marker
            
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to create marker pattern: {e}")
    #         # Fallback: simple black square with white inner square
    #         fallback = np.zeros((self.marker_size, self.marker_size, 3), dtype=np.uint8)
    #         cv2.rectangle(fallback, (5, 5), (self.marker_size-5, self.marker_size-5), (255, 255, 255), -1)
    #         return fallback
    
    def project_pattern(self, image, center_u, center_v, marker):
        """Blend marker onto image at position and 70 pixels above."""
        h, w = marker.shape[:2]
        half_h, half_w = h // 2, w // 2
        
        # Original position
        y1 = max(0, center_v - half_h)
        y2 = min(image.shape[0], center_v + half_h)
        x1 = max(0, center_u - half_w)
        x2 = min(image.shape[1], center_u + half_w)
        
        m_y1 = max(0, half_h - (center_v - y1))
        m_y2 = min(h, half_h + (y2 - center_v))
        m_x1 = max(0, half_w - (center_u - x1))
        m_x2 = min(w, half_w + (x2 - center_u))
        
        roi = image[y1:y2, x1:x2]
        marker_region = marker[m_y1:m_y2, m_x1:m_x2]
        
        if marker_region.shape[:2] != roi.shape[:2]:
            marker_region = cv2.resize(marker_region, (roi.shape[1], roi.shape[0]))
        
        alpha = self.marker_opacity
        
        if roi.shape[0] > 10 and roi.shape[1] > 10:
            mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.float32)
            border = 3
            mask[:border, :] = 0.3
            mask[-border:, :] = 0.3
            mask[:, :border] = 0.3
            mask[:, -border:] = 0.3
            mask_3d = np.stack([mask, mask, mask], axis=2)
            alpha_adjusted = alpha * mask_3d
        else:
            alpha_adjusted = alpha
        
        blended = roi * (1 - alpha_adjusted) + marker_region * alpha_adjusted
        blended = blended.astype(np.uint8)
        image[y1:y2, x1:x2] = blended
        
        """Second marker 70 pixels above"""
        center_v_above = center_v - 70
        if center_v_above - half_h < image.shape[0] and center_v_above + half_h > 0:
            y1_above = max(0, center_v_above - half_h)
            y2_above = min(image.shape[0], center_v_above 
                           + half_h)
            x1_above = max(0, center_u - half_w)
            x2_above = min(image.shape[1], center_u + half_w)
            
            m_y1_above = max(0, half_h - (center_v_above - y1_above))
            m_y2_above = min(h, half_h + (y2_above - center_v_above))
            m_x1_above = max(0, half_w - (center_u - x1_above))
            m_x2_above = min(w, half_w + (x2_above - center_u))
            
            roi_above = image[y1_above:y2_above, x1_above:x2_above]
            marker_region_above = marker[m_y1_above:m_y2_above, m_x1_above:m_x2_above]
            
            if marker_region_above.shape[:2] != roi_above.shape[:2]:
                marker_region_above = cv2.resize(marker_region_above, (roi_above.shape[1], roi_above.shape[0]))
            
            if roi_above.shape[0] > 10 and roi_above.shape[1] > 10:
                mask_above = np.ones((roi_above.shape[0], roi_above.shape[1]), dtype=np.float32)
                mask_above[:border, :] = 0.3
                mask_above[-border:, :] = 0.3
                mask_above[:, :border] = 0.3
                mask_above[:, -border:] = 0.3
                mask_3d_above = np.stack([mask_above, mask_above, mask_above], axis=2)
                alpha_adjusted_above = alpha * mask_3d_above
            else:
                alpha_adjusted_above = alpha
            
            blended_above = roi_above * (1 - alpha_adjusted_above) + marker_region_above * alpha_adjusted_above
            blended_above = blended_above.astype(np.uint8)
            image[y1_above:y2_above, x1_above:x2_above] = blended_above

def main(args=None):
    rclpy.init(args=args)
    node = VisualAugmentor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
        node.get_logger().info(f"Final stats: {node.stats}")
        node.get_logger().info(f"Log saved to: {node.log_file_path}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()