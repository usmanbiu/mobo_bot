#using 10 dinstinct geometric patterns based on fov norm intensity. note that outter background of the marker has same pattern as the main

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
import os
from datetime import datetime



class VisualAugmentor(Node):
    """
    Augments camera images with synthetic AprilTag markers at reflectivity landmark locations.
    Uses 4 distinct AprilTag families based on normalized intensity values.
    """
    
    def __init__(self):
        super().__init__('visual_augmentor')

        # Enable ALL debug logging
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # Create subscribers for synchronization
        self.image_sub = Subscriber(self, Image, '/camera_optical/image')
        self.scan_sub = Subscriber(self, LaserScan, '/lidar/scan')
        
        # Set up approximate time synchronizer
        # slop=0.1 means messages within 100ms of each other are considered synchronized
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
        
        # Lidar parameters
        self.lidar_frame = 'lidar'  # Adjust based on your TF tree
        
        self.sensor_height = 0 #0.15  #height of lidar scan from ground

        self.GLOBAL_MIN_INTENSITY = 0.0
        self.GLOBAL_MAX_INTENSITY = 255.0  # based on sensor spec

        # ========== DYNAMIC INTENSITY NORMALIZATION ==========
        # Track observed intensity range for better normalization
        self.observed_min_intensity = float('inf')
        self.observed_max_intensity = float('-inf')
        self.intensity_samples = []  # Store recent intensity samples for adaptive mapping
        self.max_samples = 500  # Keep last 1000 samples
        # =====================================================

        #map frame
        self.map_frame = 'map'
  
        #lidar field of view
        fov_deg = 140.0
        self.half_fov_rad = np.deg2rad(fov_deg / 2.0)  # ≈ 1.2217 rad

        # Current state
        self.marker_db = {}  # Stores marker patterns for consistency
        self.latest_image = None  # Store latest image for sync processing
        self.latest_image_header = None
        
        # ========== APRILTAG CONFIGURATION ==========
        # Define the 4 AprilTag families we'll use (ordered by increasing complexity)
        self.april_tag_families = [
            'TAG16H5',    # Family 0: Smallest, simplest (16x16 grid, 5 bits)
            'TAG25H7',    # Family 1: Medium-small (25x25 grid, 7 bits)
            'TAG25H9',    # Family 2: Medium-large (25x25 grid, 9 bits)  
            'TAG36H11'    # Family 3: Largest, most complex (36x36 grid, 11 bits)
        ]
        
        # Marker parameters
        self.marker_size = 40  # pixels
        self.marker_opacity = 0.9  # Blend with original image

        # For each family, we'll generate multiple tag IDs (0-9) for variety
        self.tags_per_family = 4  # IDs 0-9 available for each family  # each family has 10 tag ids that are accessed using cluster size
        
        # Cache for pre-generated AprilTag patterns
        self.april_tag_cache = {}
        self.pre_generate_pattern()
        # ============================================
        
        # CRITICAL: Intensity parameters for 0-255 range (but typically <50)
        self.reflectivity_threshold = 0.5  # Absolute threshold in 0-255 range
        self.relative_threshold_multiplier = 1.1  # Times max intensity in scan
        self.min_cluster_size = 3  # Minimum points to form a landmark
        
        
        # ========== LANDMARK LIFECYCLE MANAGEMENT ==========
        # Store confirmed and candidate landmarks
        self.landmarks = {}  # Dictionary of all tracked landmarks by ID
        self.current_scan_landmarks = []  # Landmarks from current scan (candidates)
        
        # Confirmation parameters
        self.required_observations = 10  # N = 10 observations needed for confirmation
        self.spatial_consistency_threshold = 0.3  # meters - max distance for same landmark
        self.max_landmark_age = 1.0  # seconds - remove landmarks not seen for this long
        
        # Statistics for confirmation tracking
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
        # ===================================================
        
        # ========== INTENSITY MAPPING LOGGING ==========
        self.log_file_path = self.create_log_file()
        self.log_intensity_mapping_header()
        self.mapping_counter = 0
        self.log_interval = 1  # Log every landmark (set to 1 to log all)
        # ================================================
        
        self.get_logger().info("Visual Augmentor Initialized with AprilTag Landmark Markers")
        self.get_logger().info(f"Using 4 AprilTag families: {', '.join(self.april_tag_families)}")
        self.get_logger().info(f"Confirmation requires: {self.required_observations} observations within {self.spatial_consistency_threshold}m")
        self.get_logger().info(f"Threshold: {self.reflectivity_threshold} (absolute), "
                              f"{self.relative_threshold_multiplier}x max (relative)")
        self.get_logger().info(f"Time synchronizer active: slop=0.1s, queue_size=10")
        self.get_logger().info(f"Intensity mapping log file: {self.log_file_path}")
      
    
    def create_log_file(self):
        """Create a unique log file with timestamp."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.expanduser('~/april_tag_logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"intensity_mapping_{timestamp}.csv"
        filepath = os.path.join(log_dir, filename)
        
        return filepath
    
    def log_intensity_mapping_header(self):
        """Write header to the log file."""
        try:
            with open(self.log_file_path, 'w') as f:
                f.write("timestamp,intensity_raw,intensity_normalized,intensity_idx,quarter,family,cluster_size,x_position,y_position,marker_id,fov_observed_min,fov_observed_max\n")
            self.get_logger().info(f"Created log file: {self.log_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to create log file: {e}")
    
    def log_intensity_mapping(self, landmark, intensity_idx, quarter, family, marker_id, intensity_normalized):
        """Log intensity mapping data to file."""
        try:
            self.mapping_counter += 1
            
            # Only log every N landmarks to avoid excessive file size
            if self.mapping_counter % self.log_interval != 0:
                return
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            with open(self.log_file_path, 'a') as f:
                f.write(f"{timestamp},"
                       f"{landmark['intensity']:.2f},"
                       f"{landmark['intensity_normalized']:.4f},"  # Global normalization
                       #f"{normalized_adaptive:.4f},"  # Adaptive normalization
                       f"{intensity_idx},"
                       f"{quarter},"
                       f"{family},"
                       f"{landmark.get('cluster_size', 0)},"
                       f"{landmark.get('x', 0):.3f},"
                       f"{landmark.get('y', 0):.3f},"
                       f"{marker_id},"
                       f"{self.fov_observed_min_intensity if self.fov_observed_min_intensity != float('inf') else 0:.2f},"
                       f"{self.fov_observed_max_intensity if self.fov_observed_max_intensity != float('-inf') else 255:.2f}\n")
        except Exception as e:
            self.get_logger().error(f"Failed to write to log file: {e}")
    
    def camera_info_callback(self, msg):
        """Store camera calibration parameters and image dimensions."""
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)
        self.camera_frame = "camera_optical"  # msg.header.frame_id
        self.image_width = msg.width
        self.image_height = msg.height
        self.get_logger().info(f"Camera calibration received: {self.image_width}x{self.image_height}")
    
    def sync_callback(self, image_msg, scan_msg):
        """
        Synchronized callback that receives time-aligned image and laser scan.
        This is the primary processing pipeline.
        """
        self.stats['sync_calls'] += 1
        
        # Store latest data
        self.latest_image = image_msg
        self.latest_image_header = image_msg.header
        
        # Process scan to extract landmarks in map frame (candidates for this scan)
        self.current_scan_landmarks = self.extract_high_reflectivity_landmarks(scan_msg)
        
        # ========== UPDATE LANDMARK DATABASE WITH NEW OBSERVATIONS ==========
        self.update_landmark_database()
        # ====================================================================
        
        # Get ONLY confirmed landmarks for visualization
        confirmed_landmarks = self.get_confirmed_landmarks()
        
        # Log sync stats periodically
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
        
        # Process the synchronized pair with ONLY confirmed landmarks
        self.process_synchronized_data(image_msg, confirmed_landmarks)
    
    def update_landmark_database(self):
        """
        Update the persistent landmark database with new observations from current scan.
        Implements confirmation logic requiring multiple consistent observations.
        """
        current_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
        
        # First, mark all existing landmarks as not seen in this scan
        for landmark_id in self.landmarks:
            self.landmarks[landmark_id]['seen_in_current_scan'] = False
        
        # Process each new candidate landmark from current scan
        for candidate in self.current_scan_landmarks:
            matched = False
            
            # Try to match with existing landmarks
            for landmark_id, existing in self.landmarks.items():
                # Calculate spatial distance
                dx = candidate['x'] - existing['x_map']
                dy = candidate['y'] - existing['y_map']
                dz = candidate['z'] - existing['z_map']
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                # If within threshold, it's the same landmark
                if distance < self.spatial_consistency_threshold:
                    # Update existing landmark
                    self.landmarks[landmark_id]['observation_count'] += 1
                    self.landmarks[landmark_id]['last_seen'] = current_time
                    self.landmarks[landmark_id]['seen_in_current_scan'] = True
                    
                    # Update position (running average for stability)
                    alpha = 0.3  # Weight for new observation
                    self.landmarks[landmark_id]['x_map'] = (1-alpha) * existing['x_map'] + alpha * candidate['x']
                    self.landmarks[landmark_id]['y_map'] = (1-alpha) * existing['y_map'] + alpha * candidate['y']
                    self.landmarks[landmark_id]['z_map'] = (1-alpha) * existing['z_map'] + alpha * candidate['z']
                    
                    # Update intensity (max tends to be most reliable)
                    self.landmarks[landmark_id]['intensity'] = max(
                        existing['intensity'], candidate['intensity']
                    )
                    self.landmarks[landmark_id]['intensity_normalized'] = max(
                        existing['intensity_normalized'], candidate['intensity_normalized']
                    )
                    
                    # Check if this observation pushes it to confirmed status
                    if not existing['confirmed'] and self.landmarks[landmark_id]['observation_count'] >= self.required_observations:
                        self.landmarks[landmark_id]['confirmed'] = True
                        self.stats['landmarks_confirmed'] += 1
                        self.get_logger().info(f"Landmark {landmark_id} CONFIRMED after {self.landmarks[landmark_id]['observation_count']} observations")
                    
                    matched = True
                    break
            
            # If no match found, create new landmark entry
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
                self.get_logger().debug(f"New candidate landmark: {landmark_id}")
        
        # Remove landmarks that haven't been seen for too long
        expired_ids = []
        for landmark_id, landmark in self.landmarks.items():
            if current_time - landmark['last_seen'] > self.max_landmark_age:
                expired_ids.append(landmark_id)
        
        for landmark_id in expired_ids:
            if self.landmarks[landmark_id]['confirmed']:
                self.get_logger().info(f"Confirmed landmark {landmark_id} expired (not seen for {self.max_landmark_age}s)")
            del self.landmarks[landmark_id]
            self.stats['landmarks_expired'] += 1
    
    def get_confirmed_landmarks(self):
        """
        Return list of all confirmed landmarks in the current format expected by process_synchronized_data.
        """
        confirmed_list = []
        for landmark_id, landmark in self.landmarks.items():
            if landmark['confirmed']:
                # Convert to format expected by process_synchronized_data
                confirmed_list.append({
                    'x': landmark['x_map'],
                    'y': landmark['y_map'],
                    'z': landmark['z_map'],
                    'intensity': landmark['intensity'],
                    'intensity_normalized': landmark['intensity_normalized'],
                    'cluster_size': landmark['cluster_size'],
                    'observation_count': landmark['observation_count'],
                    'landmark_id': landmark_id
                })
        return confirmed_list
    
    def generate_landmark_id(self, landmark):
        """Generate a unique ID for a new landmark based on its position."""
        # Quantize position to 10cm grid for stable IDs
        grid_size = 0.1
        x_idx = int(landmark['x'] / grid_size)
        y_idx = int(landmark['y'] / grid_size)
        z_idx = int(landmark['z'] / grid_size)
        return f"lm_{x_idx}_{y_idx}_{z_idx}"
    
    def process_synchronized_data(self, image_msg, landmarks):
        """
        Process time-synchronized image and CONFIRMED landmarks only.
        """
        if self.K is None:
            self.stats['sync_skipped_no_calib'] += 1
            self.get_logger().warn("No camera calibration yet, skipping augmentation")
            self.pub_augmented.publish(image_msg)
            return
        
        if not landmarks:
            self.stats['sync_skipped_no_landmarks'] += 1
            self.pub_augmented.publish(image_msg)
            return
        
        # Count confirmed landmarks for stats
        confirmed_count = len([l for l in self.landmarks.values() if l['confirmed']])
        if confirmed_count == 0:
            self.stats['sync_skipped_no_confirmed'] += 1
            self.pub_augmented.publish(image_msg)
            return
        
        try:
            # Convert to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            augmented = cv_image.copy()
            debug = cv_image.copy()
            
            # Process each CONFIRMED landmark (only stable ones generate markers)
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
                self.get_logger().debug("No confirmed landmarks projected to image")
                self.pub_augmented.publish(image_msg)
                return
            
            # Sort by normalized intensity (strongest first)
            valid_landmarks.sort(key=lambda x: x['landmark']['intensity_normalized'], reverse=True)
            
            # Limit number of markers to avoid clutter
            max_markers = min(5, len(valid_landmarks))
            
            for i in range(max_markers):
                data = valid_landmarks[i]
                u, v = data['uv']
                landmark = data['landmark']
                
                # Create or retrieve marker
                marker_pattern = self.get_marker_pattern(data['marker_id'], landmark)
                
                # Blend marker onto image
                self.blend_marker(augmented, u, v, marker_pattern)
                
                # Draw debug visualization with confirmation info
                intensity_norm = landmark['intensity_normalized']
                color_intensity = int(intensity_norm * 255)
                
                # Confirmed landmarks get a special border
                # Color gradient: blue (low) -> green (medium) -> red (high)
                if intensity_norm < 0.33:
                    color = (255, int(color_intensity * 3), 0)  # Blue to cyan
                elif intensity_norm < 0.66:
                    color = (255 - int(color_intensity * 1.5), 255, 0)  # Cyan to green
                else:
                    color = (0, 255 - int(color_intensity * 0.5), color_intensity)  # Green to red
                
                # Draw circle with double border for confirmed landmarks
                cv2.circle(debug, (u, v), 8, color, 2)
                cv2.circle(debug, (u, v), 10, (255, 255, 255), 1)  # White outer ring for confirmed
                
                # Add observation count to debug text
                obs_count = landmark.get('observation_count', self.required_observations)
                cv2.putText(debug, f"{landmark['intensity']:.0f}({obs_count})", 
                           (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
                
                markers_added += 1
            
            # Publish augmented image
            augmented_msg = self.bridge.cv2_to_imgmsg(augmented, encoding='bgr8')
            augmented_msg.header = image_msg.header
            self.pub_augmented.publish(augmented_msg)
            
            # Publish debug visualization with statistics
            debug_stats = (
                f"Sync #{self.stats['sync_calls']}: {markers_added}/{len(valid_landmarks)} confirmed | "
                f"Total: {len(self.landmarks)} ({confirmed_count} confirmed)"
            )
            cv2.putText(debug, debug_stats, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.pub_debug.publish(debug_msg)
            
            self.get_logger().debug(
                f"Added {markers_added} synthetic markers from {len(landmarks)} confirmed landmarks",
                throttle_duration_sec=1.0
            )
            
        except Exception as e:
            self.get_logger().error(f"Augmentation failed: {e}")
            # Pass through original image on error
            self.pub_augmented.publish(image_msg)
        
    
    def extract_high_reflectivity_landmarks(self, scan_msg):
        """
        Extract and cluster high reflectivity points from LaserScan.
        """
        ranges = np.array(scan_msg.ranges)
        intensities = np.array(scan_msg.intensities, dtype=np.float32)
        
        self.stats['total_points'] += len(intensities)
        
        if len(intensities) == 0:
            return []
        
        # Create angle array
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        
        # ========== STEP 1: BASIC FILTERS (RANGE & FINITE) FOR NORMALIZATION ==========
        range_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        finite_mask = np.isfinite(intensities)
        valid_for_norm_mask = range_mask & finite_mask
        
        # Get intensities for normalization (using full 360° data, no FOV filter yet)
        intensities_all = intensities[valid_for_norm_mask]
        
        if len(intensities_all) == 0:
            return []
        
        # Calculate min/max from FULL 360° scan
        min_intensity_full = np.min(intensities_all)  #min intensities for raw intensities 
        max_intensity_full = np.max(intensities_all)   #max intensities for raw intensities 
        
        # Normalize intensities using full range
        if max_intensity_full > min_intensity_full:
            norm_intensities_full = (intensities_all - min_intensity_full) / (max_intensity_full - min_intensity_full)
        else:
            norm_intensities_full = np.zeros_like(intensities_all)
        
        self.get_logger().debug(
            f"Full scan intensity range: {min_intensity_full:.2f} - {max_intensity_full:.2f}",
            throttle_duration_sec=2.0
        )
        
        # ========== STEP 2: APPLY FOV FILTER ==========
        forward_fov_mask = (angles >= -self.half_fov_rad) & (angles <= self.half_fov_rad)
        final_mask = valid_for_norm_mask & forward_fov_mask
        
        # Extract filtered data
        valid_ranges = ranges[final_mask]
        valid_intensities = intensities[final_mask]
        valid_angles = angles[final_mask]
        
        # Get corresponding normalized intensities
        valid_norm_indices = np.where(valid_for_norm_mask)[0]
        fov_filtered_indices = valid_norm_indices[forward_fov_mask[valid_for_norm_mask]]
        
        if len(fov_filtered_indices) > 0:
            fov_norm_intensities = norm_intensities_full[forward_fov_mask[valid_for_norm_mask]]
        else:
            fov_norm_intensities = np.array([])
        
        if len(valid_ranges) == 0:
            return []  
        
        median_fov_intensity = np.median(fov_norm_intensities)
        
        # Dynamic threshold calculation
        absolute_threshold = self.reflectivity_threshold
        relative_threshold = median_fov_intensity * self.relative_threshold_multiplier
        threshold = max(absolute_threshold, relative_threshold)
        
        self.get_logger().debug(
            f"Intensity thresholds: median={median_fov_intensity:.3f}, "
            f"abs_thresh={absolute_threshold:.3f}, "
            f"rel_thresh={relative_threshold:.3f}, "
            f"final={threshold:.3f}",
            throttle_duration_sec=2.0
        )
        
        # Find high reflectivity points
        high_reflectivity_mask = fov_norm_intensities > threshold
        self.stats['high_reflectivity_points'] += np.sum(high_reflectivity_mask)
        
        high_ranges = valid_ranges[high_reflectivity_mask]
        high_intensities_norm = fov_norm_intensities[high_reflectivity_mask]
        high_intensities_raw = valid_intensities[high_reflectivity_mask]  # Keep raw for logging
        high_angles = valid_angles[high_reflectivity_mask]

        self.fov_observed_min_intensity = np.min(valid_intensities)  #max intensities for intensities in the FOV
        self.fov_observed_max_intensity = np.max(valid_intensities)   #min intensities for intensities in the FOV
        
        if len(high_ranges) == 0:
            return []
        
        # Convert to Cartesian coordinates
        x = high_ranges * np.cos(high_angles)
        y = high_ranges * np.sin(high_angles)
        z = np.full_like(x, self.sensor_height)
        
        # ========== FIX: GET TRANSFORM USING SCAN TIMESTAMP ==========
        try:
            # Use the scan's timestamp for TF lookup
            transform_l_m = self.tf_buffer.lookup_transform(
                self.map_frame,      # target frame
                self.lidar_frame,    # source frame
                scan_msg.header.stamp,  # Use the scan's timestamp!
                rclpy.duration.Duration(seconds=0.1)  # Allow 100ms tolerance
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup at time {scan_msg.header.stamp} failed: {e}")
            
            # Fallback: try the latest transform
            try:
                self.get_logger().info("Attempting fallback with latest transform")
                transform_l_m = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.lidar_frame,
                    rclpy.time.Time()  # Latest available
                )
            except Exception as e2:
                self.get_logger().error(f"Fallback TF lookup also failed: {e2}")
                return []
        
        # ========== CREATE LANDMARKS ==========
        landmarks = []
        processed = np.zeros(len(x), dtype=bool)
        
        for i in range(len(x)):
            if processed[i]:
                continue
            
            # Adaptive cluster radius based on distance
            distance = np.sqrt(x[i]**2 + y[i]**2)
            cluster_radius = 0.3 + 0.05 * (distance / 5.0)
            
            # Find points close to this one
            distances = np.sqrt((x - x[i])**2 + (y - y[i])**2)
            cluster_mask = distances < cluster_radius
            
            cluster_size = np.sum(cluster_mask)
            if cluster_size >= self.min_cluster_size:
                cluster_x = np.max(x[cluster_mask])
                cluster_y = np.max(y[cluster_mask])
                cluster_z = np.max(z[cluster_mask])
                cluster_intensity_norm = np.mean(high_intensities_norm[cluster_mask])
                cluster_intensity_raw = np.mean(high_intensities_raw[cluster_mask])
                
                # Transform to map frame
                p_lidar = PointStamped()
                p_lidar.header.frame_id = self.lidar_frame
                p_lidar.header.stamp = scan_msg.header.stamp  # Important: set the timestamp!
                p_lidar.point.x = cluster_x
                p_lidar.point.y = cluster_y
                p_lidar.point.z = cluster_z
                
                p_map = do_transform_point(p_lidar, transform_l_m)
                
                landmarks.append({
                    'x': p_map.point.x,
                    'y': p_map.point.y,
                    'z': p_map.point.z,
                    'intensity': float(cluster_intensity_raw),  # Raw intensity for logging
                    'intensity_normalized': float(cluster_intensity_norm),  # Normalized for pattern selection
                    'cluster_size': 1,
                    'distance': float(distance),
                    'raw_points': list(zip(x[cluster_mask], y[cluster_mask]))
                })
                
                processed[cluster_mask] = True
                self.stats['landmarks_created'] += 1
        
        return landmarks
   #project image function for using lidar-map-camera transform (anchoring lidar landmarks to map)
    def project_to_image(self, landmark):
        """
        Project a world-anchored LiDAR landmark into camera image pixels.
        Pipeline: MAP → CAMERA_OPTICAL → IMAGE
        """
        if self.K is None:
            self.get_logger().warn("Camera intrinsics not available")
            return None

        try:
            # ========== FIX: Use the current time or the landmark's timestamp ==========
            # Since we don't have a specific timestamp for the landmark,
            # we use Time(0) to get the latest transform, but with a timeout
            transform_m_c = self.tf_buffer.lookup_transform(
                self.camera_frame,  # target
                self.map_frame,     # source
                rclpy.time.Time(),  # Latest available transform
                rclpy.duration.Duration(seconds=0.1)  # 100ms timeout
            )
            
            p_map = PointStamped()
            p_map.header.frame_id = self.map_frame
            p_map.point.x = landmark["x"]
            p_map.point.y = landmark["y"]
            p_map.point.z = landmark["z"]
            
            # Transform landmarks to camera frame
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
                # Optional: log out-of-bounds only occasionally
                if np.random.rand() < 0.01:  # Log ~1% of out-of-bounds
                    self.get_logger().debug(f"Point outside image: ({u}, {v})")
                return None

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF error during projection: {e}")
            return None
    # ========== APRILTAG GENERATION METHODS ==========
    
    def pre_generate_pattern(self):
        """
        Pre-generate all polygon patterns for the 10-tag mapping.
        Each pattern is a white square with a black polygon inside.
        """
        try:
            self.april_tag_available = True
            
            # Generate all 10 patterns (tag_id 0-9)
            for tag_id in range(10):
                cache_key = f"polygon_{tag_id}"
                
                # Generate the polygon pattern
                tag_pattern = self.generate_pattern("POLYGON", tag_id)
                
                if tag_pattern is not None:
                    self.april_tag_cache[cache_key] = tag_pattern
                    self.get_logger().debug(f"Generated polygon pattern ID:{tag_id}")
            
            self.get_logger().info(f"Pre-generated {len(self.april_tag_cache)} polygon patterns")
            
        except Exception as e:
            self.april_tag_available = False
            self.get_logger().warn(f"Polygon pattern generation failed: {e}")
            self.get_logger().warn("Falling back to fallback pattern generator")
    

    
    def generate_pattern(self, family, tag_id):
        """
        Generate a simple pattern: white square with a black square inside.
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
    
    def get_fallback_pattern(self, intensity_quarter):
        """
        Generate a fallback pattern if AprilTag generation fails.
        Creates simple geometric patterns based on intensity quarter.
        
        Args:
            intensity_quarter: 0, 1, 2, or 3
            
        Returns:
            numpy array: BGR image of fallback pattern
        """
        size = self.marker_size
        pattern = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Different patterns for each quarter
        if intensity_quarter == 0:
            # Quarter 0: Circle
            cv2.circle(pattern, (size//2, size//2), size//3, (255, 255, 255), -1)
            cv2.circle(pattern, (size//2, size//2), size//4, (0, 0, 0), -1)
            
        elif intensity_quarter == 1:
            # Quarter 1: Square with cross
            cv2.rectangle(pattern, (size//4, size//4), (3*size//4, 3*size//4), (255, 255, 255), -1)
            cv2.line(pattern, (size//4, size//4), (3*size//4, 3*size//4), (0, 0, 0), 2)
            cv2.line(pattern, (size//4, 3*size//4), (3*size//4, size//4), (0, 0, 0), 2)
            
        elif intensity_quarter == 2:
            # Quarter 2: Triangle
            pts = np.array([[size//2, size//4], 
                           [size//4, 3*size//4], 
                           [3*size//4, 3*size//4]], np.int32)
            cv2.fillPoly(pattern, [pts], (255, 255, 255))
            cv2.polylines(pattern, [pts], True, (0, 0, 0), 2)
            
        else:  # quarter 3
            # Quarter 3: Checkerboard
            cell_size = size // 4
            for i in range(4):
                for j in range(4):
                    if (i + j) % 2 == 0:
                        color = (255, 255, 255)
                    else:
                        color = (0, 0, 0)
                    y1, y2 = i*cell_size, (i+1)*cell_size
                    x1, x2 = j*cell_size, (j+1)*cell_size
                    pattern[y1:y2, x1:x2] = color
        
        return pattern
    
    def get_marker_id(self, landmark):
        """Generate consistent marker ID for a landmark with intensity mapping."""
        # Use quantized position and intensity as ID
        grid_size = 0.1  # 10cm grid
        
        # Quantize position
        x_idx = int(landmark['x'] / grid_size)
        y_idx = int(landmark['y'] / grid_size)
        
        # Use ADAPTIVE normalization for intensity mapping
        # This will spread your actual intensity range across all quarters
        if 'intensity_normalized' in landmark:
            intensity_norm = landmark['intensity_normalized']  #originally intensity_normalized_adaptive, but we are using norm intensity from start in landmark.intensity see line 760ish
        else:
            # Fallback to intensity if adaptive not available (now normalized with recent changes)
            intensity_norm = landmark['intensity']
        
        # Quantize intensity to 0-9 scale
        intensity_idx = min(9, int(intensity_norm * 10))
        
        # Include cluster size for uniqueness
        cluster_idx = min(9, landmark['cluster_size'])
        
        return f"{x_idx}_{y_idx}_{intensity_idx}_{cluster_idx}"
        
    def get_marker_pattern(self, marker_id, landmark=None):
        """
        Get or create polygon marker pattern with 10 total tags.
        Maps 10 intensity levels to 10 specific polygon patterns.
        """
        if marker_id not in self.marker_db:
            # Parse intensity from marker_id (format: "x_y_intensity_cluster")
            parts = marker_id.split('_')
            if len(parts) >= 3:
                try:
                    intensity_idx = int(parts[2])  # This is 0-9 from get_marker_id
                    
                    # Use intensity_idx directly as tag_id (0-9)
                    tag_id = intensity_idx
                    
                    # Cache key for this specific pattern
                    cache_key = f"polygon_{tag_id}"
                    
                    # Calculate quarter for logging (0-3)
                    if intensity_idx <= 2:
                        quarter = 0
                    elif intensity_idx <= 5:
                        quarter = 1
                    elif intensity_idx <= 8:
                        quarter = 2
                    else:
                        quarter = 3
                    
                    # Try to get pattern from cache
                    if cache_key in self.april_tag_cache:
                        pattern = self.april_tag_cache[cache_key]
                    elif hasattr(self, 'april_tag_available') and self.april_tag_available:
                        # Generate on-the-fly if not in cache
                        pattern = self.generate_pattern("POLYGON", tag_id)
                        if pattern is not None:
                            self.april_tag_cache[cache_key] = pattern
                        else:
                            # Fallback to geometric pattern if generation fails
                            pattern = self.get_fallback_pattern(quarter)
                    else:
                        pattern = self.get_fallback_pattern(quarter)
                    
                    # ========== LOG THE INTENSITY MAPPING ==========
                    if landmark is not None:
                        # Get the intensity normalized value
                        intensity_normalized = landmark.get('intensity_normalized', landmark['intensity'])
                        self.log_intensity_mapping(
                            landmark, 
                            intensity_idx, 
                            quarter, 
                            f"polygon_{tag_id}", 
                            marker_id, 
                            intensity_normalized
                        )
                    # ================================================
                    
                    self.get_logger().debug(
                        f"Marker ID: {marker_id} | Intensity: {intensity_idx} -> "
                        f"Quarter: {quarter} | Polygon: {tag_id}",
                        throttle_duration_sec=1.0
                    )
                    
                except Exception as e:
                    self.get_logger().warn(f"Error generating polygon pattern: {e}, using fallback")
                    quarter = int(hashlib.md5(marker_id.encode()).hexdigest(), 16) % 4
                    pattern = self.get_fallback_pattern(quarter)
            else:
                # Fallback if marker_id format is unexpected
                quarter = int(hashlib.md5(marker_id.encode()).hexdigest(), 16) % 4
                pattern = self.get_fallback_pattern(quarter)
            
            # Store in database
            self.marker_db[marker_id] = pattern
            
            # Keep DB size manageable (limit to 100 entries)
            if len(self.marker_db) > 100:
                oldest_key = next(iter(self.marker_db))
                del self.marker_db[oldest_key]
        
        return self.marker_db[marker_id]    
    
    def blend_marker(self, image, center_u, center_v, marker):
        """Blend marker pattern onto image at specified location and 70 pixels above."""
        h, w = marker.shape[:2]                    # Get marker height and width
        half_h, half_w = h // 2, w // 2            # Calculate half dimensions for centering
        
        # ===== FIRST MARKER - Original position =====
        # Calculate ROI bounds for original marker
        y1 = max(0, center_v - half_h)             # Top bound - prevent going above image
        y2 = min(image.shape[0], center_v + half_h) # Bottom bound - prevent going below image
        x1 = max(0, center_u - half_w)             # Left bound - prevent going left of image
        x2 = min(image.shape[1], center_u + half_w) # Right bound - prevent going right of image
        
        # Calculate corresponding marker region for original marker
        m_y1 = max(0, half_h - (center_v - y1))    # Top of marker to use (if cropped)
        m_y2 = min(h, half_h + (y2 - center_v))    # Bottom of marker to use (if cropped)
        m_x1 = max(0, half_w - (center_u - x1))    # Left of marker to use (if cropped)
        m_x2 = min(w, half_w + (x2 - center_u))    # Right of marker to use (if cropped)
        
        # Extract regions for original marker
        roi = image[y1:y2, x1:x2]                  # Extract Region of Interest from image
        marker_region = marker[m_y1:m_y2, m_x1:m_x2] # Extract corresponding part of marker
        
        # Ensure same size for original marker
        if marker_region.shape[:2] != roi.shape[:2]:
            marker_region = cv2.resize(marker_region, 
                                    (roi.shape[1], roi.shape[0]))
        
        # Alpha blending with edge feathering for original marker
        alpha = self.marker_opacity                 # Get base opacity (typically 0.7)
        
        # Create soft mask for smoother blending if region is large enough
        if roi.shape[0] > 10 and roi.shape[1] > 10:
            # Create Gaussian mask for feathering
            mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.float32)
            border = 3
            mask[:border, :] = 0.3                   # Top border - more transparent
            mask[-border:, :] = 0.3                   # Bottom border - more transparent
            mask[:, :border] = 0.3                    # Left border - more transparent
            mask[:, -border:] = 0.3                   # Right border - more transparent
            
            # Expand to 3 channels
            mask_3d = np.stack([mask, mask, mask], axis=2)
            alpha_adjusted = alpha * mask_3d
        else:
            alpha_adjusted = alpha
        
        # Blend original marker
        blended = roi * (1 - alpha_adjusted) + marker_region * alpha_adjusted
        blended = blended.astype(np.uint8)
        
        # Copy back to image for original marker
        image[y1:y2, x1:x2] = blended
        
        # ===== SECOND MARKER - 70 pixels above =====
        center_v_above = center_v - 70              # Move marker up by 70 pixels
        
        # Only place second marker if it would be visible (within image bounds)
        if center_v_above - half_h < image.shape[0] and center_v_above + half_h > 0:
            
            # Calculate ROI bounds for marker above
            y1_above = max(0, center_v_above - half_h)
            y2_above = min(image.shape[0], center_v_above + half_h)
            x1_above = max(0, center_u - half_w)    # Same horizontal position
            x2_above = min(image.shape[1], center_u + half_w)
            
            # Calculate corresponding marker region for marker above
            m_y1_above = max(0, half_h - (center_v_above - y1_above))
            m_y2_above = min(h, half_h + (y2_above - center_v_above))
            m_x1_above = max(0, half_w - (center_u - x1_above))
            m_x2_above = min(w, half_w + (x2_above - center_u))
            
            # Extract regions for marker above
            roi_above = image[y1_above:y2_above, x1_above:x2_above]
            marker_region_above = marker[m_y1_above:m_y2_above, m_x1_above:m_x2_above]
            
            # Ensure same size for marker above
            if marker_region_above.shape[:2] != roi_above.shape[:2]:
                marker_region_above = cv2.resize(marker_region_above, 
                                            (roi_above.shape[1], roi_above.shape[0]))
            
            # Create feathering mask for marker above
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
            
            # Blend marker above
            blended_above = roi_above * (1 - alpha_adjusted_above) + marker_region_above * alpha_adjusted_above
            blended_above = blended_above.astype(np.uint8)
            
            # Copy back to image for marker above
            image[y1_above:y2_above, x1_above:x2_above] = blended_above
            
            self.get_logger().debug(f"Placed second marker at ({center_u}, {center_v_above})", 
                                throttle_duration_sec=1.0)
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
                              f"Landmarks={node.stats['landmarks_created']}, "
                              f"SyncCalls={node.stats['sync_calls']}, "
                              f"SkippedNoCalib={node.stats['sync_skipped_no_calib']}, "
                              f"SkippedNoLandmarks={node.stats['sync_skipped_no_landmarks']}")
        node.get_logger().info(f"Observed intensity range: {node.observed_min_intensity:.2f} - {node.observed_max_intensity:.2f}")
        node.get_logger().info(f"Intensity mapping log saved to: {node.log_file_path}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()




        # def generate_pattern(self, family, tag_id):
        # """
        # Generate a simple pattern: white square with a black polygon inside.
        
        # Different polygons based on tag_id (0-9) for visual distinction:
        # 0: Circle
        # 1: Star (4-point)
        # 2: Plus sign
        # 3: Octagon
        # 4: Hexagon
        # 5: Diamond
        # 6: Pentagon
        # 7: Square
        # 8: Triangle
        # 9: Cross
        # """
        # try:
        #     # Create white square background
        #     marker = np.ones((self.marker_size, self.marker_size, 3), dtype=np.uint8) * 255
            
        #     # Calculate center and size
        #     center = (self.marker_size // 2, self.marker_size // 2)
        #     polygon_size = self.marker_size // 2  # Polygon takes 1/3 of marker size
            
        #     # Define polygon points based on tag_id
        #     if tag_id == 0:  # Circle
        #         cv2.circle(marker, center, polygon_size, (0, 0, 0), -1)
                
        #     elif tag_id == 7:  # Square
        #         half = polygon_size // 2
        #         top_left = (center[0] - half, center[1] - half)
        #         bottom_right = (center[0] + half, center[1] + half)
        #         cv2.rectangle(marker, top_left, bottom_right, (0, 0, 0), -1)
                
        #     elif tag_id == 8:  # Triangle (equilateral)
        #         side = polygon_size
        #         height = int(side * np.sqrt(3) / 2)
        #         points = np.array([
        #             [center[0], center[1] - height//2],
        #             [center[0] - side//2, center[1] + height//2],
        #             [center[0] + side//2, center[1] + height//2]
        #         ], np.int32)
        #         cv2.fillPoly(marker, [points], (0, 0, 0))
                
        #     elif tag_id == 5:  # Diamond (rotated square)
        #         half = polygon_size // 2
        #         points = np.array([
        #             [center[0], center[1] - half],
        #             [center[0] + half, center[1]],
        #             [center[0], center[1] + half],
        #             [center[0] - half, center[1]]
        #         ], np.int32)
        #         cv2.fillPoly(marker, [points], (0, 0, 0))
                
        #     elif tag_id == 6:  # Pentagon
        #         radius = polygon_size * 0.7
        #         points = []
        #         for i in range(5):
        #             angle = -90 + i * 72  # Start from top
        #             rad = np.deg2rad(angle)
        #             x = center[0] + int(radius * np.cos(rad))
        #             y = center[1] + int(radius * np.sin(rad))
        #             points.append([x, y])
        #         cv2.fillPoly(marker, [np.array(points, np.int32)], (0, 0, 0))
                
        #     elif tag_id == 4:  # Hexagon
        #         radius = polygon_size * 0.8
        #         points = []
        #         for i in range(6):
        #             angle = i * 60
        #             rad = np.deg2rad(angle)
        #             x = center[0] + int(radius * np.cos(rad))
        #             y = center[1] + int(radius * np.sin(rad))
        #             points.append([x, y])
        #         cv2.fillPoly(marker, [np.array(points, np.int32)], (0, 0, 0))
                
        #     elif tag_id == 1:  # 4-point Star
        #         outer_radius = polygon_size
        #         inner_radius = polygon_size // 2
        #         points = []
        #         for i in range(8):
        #             angle = i * 45
        #             rad = np.deg2rad(angle)
        #             radius = outer_radius if i % 2 == 0 else inner_radius
        #             x = center[0] + int(radius * np.cos(rad))
        #             y = center[1] + int(radius * np.sin(rad))
        #             points.append([x, y])
        #         cv2.fillPoly(marker, [np.array(points, np.int32)], (0, 0, 0))
                
        #     elif tag_id == 9:  # Cross (thick)
        #         thickness = polygon_size // 3
        #         half_h = polygon_size // 2
        #         # Vertical bar
        #         cv2.rectangle(marker, 
        #                     (center[0] - thickness//2, center[1] - half_h),
        #                     (center[0] + thickness//2, center[1] + half_h),
        #                     (0, 0, 0), -1)
        #         # Horizontal bar
        #         cv2.rectangle(marker,
        #                     (center[0] - half_h, center[1] - thickness//2),
        #                     (center[0] + half_h, center[1] + thickness//2),
        #                     (0, 0, 0), -1)
                            
        #     elif tag_id == 2:  # Plus sign (similar to cross but thinner)
        #         thickness = polygon_size // 4
        #         half_h = polygon_size // 2
        #         # Vertical bar
        #         cv2.rectangle(marker, 
        #                     (center[0] - thickness//2, center[1] - half_h),
        #                     (center[0] + thickness//2, center[1] + half_h),
        #                     (0, 0, 0), -1)
        #         # Horizontal bar
        #         cv2.rectangle(marker,
        #                     (center[0] - half_h, center[1] - thickness//2),
        #                     (center[0] + half_h, center[1] + thickness//2),
        #                     (0, 0, 0), -1)
                            
        #     elif tag_id == 3:  # Octagon
        #         radius = polygon_size
        #         points = []
        #         for i in range(8):
        #             angle = i * 45
        #             rad = np.deg2rad(angle)
        #             x = center[0] + int(radius * np.cos(rad))
        #             y = center[1] + int(radius * np.sin(rad))
        #             points.append([x, y])
        #         cv2.fillPoly(marker, [np.array(points, np.int32)], (0, 0, 0))
            
        #     # Add a thin black border around the entire marker for contrast
        #     cv2.rectangle(marker, (0, 0), (self.marker_size-1, self.marker_size-1), (0, 0, 0), 1)
            
        #     return marker
            
        # except Exception as e:
        #     self.get_logger().error(f"Failed to generate pattern for tag_id {tag_id}: {e}")
        #     # Fallback: simple black square
        #     fallback = np.zeros((self.marker_size, self.marker_size, 3), dtype=np.uint8)
        #     cv2.rectangle(fallback, (5, 5), (self.marker_size-5, self.marker_size-5), (255, 255, 255), -1)
        #     return fallback
    
