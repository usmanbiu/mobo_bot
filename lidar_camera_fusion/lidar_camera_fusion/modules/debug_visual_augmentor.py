#!/usr/bin/env python3
"""
Diagnostic tool to debug Visual Augmentor issues.
Run this while your augmentor is running to see what's happening.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Image, CameraInfo
import numpy as np
from cv_bridge import CvBridge
import cv2
import time

class AugmentorDebugger(Node):
    def __init__(self):
        super().__init__('augmentor_debugger')
        
        # QoS for sensor data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers to monitor all topics
        self.sub_scan = self.create_subscription(
            LaserScan, '/lidar/scan', self.scan_callback, qos_profile)
        
        self.sub_camera = self.create_subscription(
            Image, '/camera_optical/image_raw', self.image_callback, qos_profile)
        
        self.sub_camera_info = self.create_subscription(
            CameraInfo, '/camera_optical/camera_info', self.camera_info_callback, qos_profile)
        
        self.sub_augmented = self.create_subscription(
            Image, '/camera/image_augmented', self.augmented_callback, qos_profile)
        
        self.sub_debug = self.create_subscription(
            Image, '/augmentation/debug', self.debug_callback, qos_profile)
        
        # OpenCV bridge
        self.bridge = CvBridge()
        
        # Statistics
        self.stats = {
            'scan_count': 0,
            'image_count': 0,
            'last_scan_time': 0,
            'last_image_time': 0,
            'has_camera_info': False,
            'intensity_stats': [],
            'landmark_count': 0
        }
        
        # TF frame checker
        self.create_timer(2.0, self.check_tf_frames)
        
        # Diagnostic timer
        self.create_timer(5.0, self.print_diagnostics)
        
        self.get_logger().info("🚀 Augmentor Debugger Started - Press Ctrl+C to stop")
    
    def check_tf_frames(self):
        """Check if required TF frames exist."""
        import tf2_ros
        try:
            tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(tf_buffer, self)
            
            # Check common frame pairs
            frames_to_check = [
                ('lidar', 'camera'),
                ('base_link', 'lidar'),
                ('base_link', 'camera'),
                ('map', 'base_link')
            ]
            
            for parent, child in frames_to_check:
                try:
                    transform = tf_buffer.lookup_transform(
                        parent, child, rclpy.time.Time())
                    self.get_logger().info(f"✅ TF: {parent} → {child} exists")
                except:
                    self.get_logger().warn(f"⚠️  TF: {parent} → {child} NOT found")
        
        except ImportError:
            self.get_logger().warn("tf2_ros not available - skipping TF check")
    
    def scan_callback(self, msg):
        """Analyze laser scan data."""
        self.stats['scan_count'] += 1
        self.stats['last_scan_time'] = time.time()
        
        # Extract intensity statistics
        if msg.intensities:
            intensities = np.array(msg.intensities)
            valid_intensities = intensities[intensities > 0]
            
            if len(valid_intensities) > 0:
                stats = {
                    'min': float(np.min(valid_intensities)),
                    'max': float(np.max(valid_intensities)),
                    'mean': float(np.mean(valid_intensities)),
                    'count': len(valid_intensities),
                    'timestamp': time.time()
                }
                self.stats['intensity_stats'].append(stats)
                
                # Keep only recent stats
                if len(self.stats['intensity_stats']) > 100:
                    self.stats['intensity_stats'].pop(0)
                
                # Log high reflectivity points
                high_reflectivity = valid_intensities[valid_intensities > 0.5]
                if len(high_reflectivity) > 0:
                    self.get_logger().info(
                        f"📡 Scan #{self.stats['scan_count']}: "
                        f"{len(high_reflectivity)} high-reflectivity points "
                        f"(max: {np.max(valid_intensities):.2f})"
                    )
        
        # Check scan properties
        if self.stats['scan_count'] % 10 == 0:
            self.get_logger().debug(
                f"Scan: {len(msg.ranges)} points, "
                f"Intensities: {'Yes' if msg.intensities else 'No'}"
            )
    
    def image_callback(self, msg):
        """Monitor camera images."""
        self.stats['image_count'] += 1
        self.stats['last_image_time'] = time.time()
        
        if self.stats['image_count'] % 10 == 0:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.get_logger().debug(
                    f"📷 Image #{self.stats['image_count']}: "
                    f"{cv_image.shape[1]}x{cv_image.shape[0]}"
                )
            except:
                pass
    
    def camera_info_callback(self, msg):
        """Check camera calibration."""
        if not self.stats['has_camera_info']:
            self.stats['has_camera_info'] = True
            self.get_logger().info(
                f"✅ Camera calibration received: "
                f"K={msg.k[0]:.0f},{msg.k[4]:.0f},{msg.k[2]:.0f},{msg.k[5]:.0f}"
            )
    
    def augmented_callback(self, msg):
        """Check augmented images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Check if augmentation happened (look for synthetic markers)
            # Simple check: look for high-contrast small patterns
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            if edge_density > 0.01:  # Arbitrary threshold
                self.get_logger().info(f"🎨 Augmented image has high edge density: {edge_density:.3f}")
        
        except Exception as e:
            self.get_logger().error(f"Failed to process augmented image: {e}")
    
    def debug_callback(self, msg):
        """Analyze debug visualization."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Look for debug text or markers
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Look for yellow markers (BGR: 0, 255, 255)
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            yellow_pixels = np.sum(yellow_mask > 0)
            
            if yellow_pixels > 10:
                self.get_logger().info(f"🔍 Debug image has {yellow_pixels} yellow pixels (markers?)")
            
            # Check image for text (simple brightness check in center)
            height, width = cv_image.shape[:2]
            center_region = cv_image[height//2-50:height//2+50, width//2-100:width//2+100]
            if center_region.size > 0:
                avg_brightness = np.mean(center_region)
                if avg_brightness > 200:  # Likely text on dark background
                    self.get_logger().info("📝 Debug image appears to have text overlay")
        
        except Exception as e:
            self.get_logger().error(f"Failed to analyze debug image: {e}")
    
    def print_diagnostics(self):
        """Print comprehensive diagnostic information."""
        print("\n" + "="*60)
        print("VISUAL AUGMENTOR DIAGNOSTICS")
        print("="*60)
        
        # Topic status
        print("\n📡 TOPIC STATUS:")
        print(f"  LaserScan received: {self.stats['scan_count']} messages")
        print(f"  Camera images received: {self.stats['image_count']} messages")
        print(f"  Camera calibration: {'✅ Yes' if self.stats['has_camera_info'] else '❌ No'}")
        
        # Time checks
        current_time = time.time()
        scan_age = current_time - self.stats['last_scan_time']
        image_age = current_time - self.stats['last_image_time']
        
        print(f"\n⏰ TIMING:")
        print(f"  Last scan: {scan_age:.1f}s ago")
        print(f"  Last image: {image_age:.1f}s ago")
        
        # Intensity analysis
        if self.stats['intensity_stats']:
            recent_stats = self.stats['intensity_stats'][-5:]  # Last 5 scans
            max_intensities = [s['max'] for s in recent_stats]
            avg_max = np.mean(max_intensities) if max_intensities else 0
            
            print(f"\n💡 INTENSITY ANALYSIS (recent):")
            print(f"  Max intensity: {avg_max:.3f}")
            print(f"  Threshold (70%): {avg_max * 0.7:.3f}")
            print(f"  Min threshold: 0.300")
            
            if avg_max > 0:
                if avg_max * 0.7 < 0.3:
                    print(f"  ⚠️  Using MIN threshold (0.3) - intensities too low!")
                else:
                    print(f"  ✅ Using RELATIVE threshold ({avg_max * 0.7:.3f})")
            else:
                print(f"  ❌ No valid intensities detected!")
        
        # TF frame check
        print(f"\n🔄 TF FRAMES:")
        print(f"  (Run 'ros2 run tf2_ros tf2_echo lidar_frame camera_frame' to check)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if self.stats['scan_count'] == 0:
            print("  1. ❌ NO LASER SCANS - Check if /lidar/scan is publishing")
            print("     Run: ros2 topic echo /lidar/scan --no-arr | head -5")
        
        if not self.stats['has_camera_info']:
            print("  2. ❌ NO CAMERA CALIBRATION - Check /camera/camera_info")
        
        if self.stats['intensity_stats']:
            if avg_max < 0.3:
                print("  3. ⚠️  LOW REFLECTIVITY - Try lowering min_intensity_absolute")
                print("     Or ensure there are reflective surfaces in environment")
        
        if scan_age > 10:
            print("  4. ⚠️  OLD SCAN DATA - Lidar may have stopped")
        
        if image_age > 10:
            print("  5. ⚠️  OLD IMAGE DATA - Camera may have stopped")
        
        print("\n" + "="*60)

def main(args=None):
    rclpy.init(args=args)
    node = AugmentorDebugger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n👋 Debugger stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()