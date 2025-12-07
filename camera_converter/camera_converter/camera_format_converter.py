#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraFormatConverter(Node):
    def __init__(self):
        super().__init__('camera_format_converter')
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Publishers (converted format)
        self.pub_image = self.create_publisher(Image, '/camera_optical/image_converted', 10)
        #self.pub_camera_info = self.create_publisher(CameraInfo, '/camera_optical/camera_info_converted', 10)
        
        # Subscribers (original format)
        self.image_sub = self.create_subscription(
            Image, '/camera_optical/image_raw', self.image_callback, 10)
        # self.camera_info_sub = self.create_subscription(
        #     CameraInfo, '/camera_optical/camera_info', self.camera_info_callback, 10)
        
        self.get_logger().info("Camera format converter started - converting YUV422 to BGR8")

    def image_callback(self, msg):
        """Convert YUV422 to BGR8 format."""
        try:
            # Convert ROS Image to OpenCV (YUV422)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Convert YUV422 to BGR
            if len(cv_image.shape) == 2:  # Already mono or unexpected format
                bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            else:
                # Convert YUV422 (YUYV) to BGR
                bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_YUV2BGR_YUYV)
            
            # Convert back to ROS Image (BGR8)
            converted_msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
            converted_msg.header = msg.header
            
            # Publish converted image
            self.pub_image.publish(converted_msg)
            
            self.get_logger().info("Converted YUV422 to BGR8", throttle_duration_sec=5.0)
            
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {str(e)}")

    # def camera_info_callback(self, msg):
    #     """Pass through camera info with updated timestamp if needed."""
    #     # Just republish with same or updated header
    #     info_msg = msg
    #     self.pub_camera_info.publish(info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraFormatConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()




#     #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CameraInfo, LaserScan
# from nav_msgs.msg import Odometry
# from cv_bridge import CvBridge
# import cv2
# import threading
# from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# class CompleteSyncNode(Node):
#     def __init__(self):
#         super().__init__('complete_sync_node')
        
#         self.bridge = CvBridge()
        
#         # Storage with locks
#         self.latest_odom = None
#         self.latest_scan = None  
#         self.latest_image = None
#         self.latest_camera_info = None
#         self.odom_lock = threading.Lock()
#         self.scan_lock = threading.Lock()
#         self.image_lock = threading.Lock()
#         self.camera_info_lock = threading.Lock()
        
#         # QoS profiles for different data types
#         best_effort_qos = QoSProfile(
#             depth=10,
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         reliable_qos = QoSProfile(
#             depth=10,
#             reliability=ReliabilityPolicy.RELIABLE,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         # Publishers - use best effort for real-time data
#         self.pub_odom = self.create_publisher(Odometry, '/sync/odom', best_effort_qos)
#         self.pub_image = self.create_publisher(Image, '/sync/image_raw', best_effort_qos)
#         self.pub_camera_info = self.create_publisher(CameraInfo, '/sync/camera_info', best_effort_qos)
#         self.pub_scan = self.create_publisher(LaserScan, '/sync/scan', best_effort_qos)
        
#         # Subscribers
#         self.odom_sub = self.create_subscription(
#             Odometry, '/odom', self.odom_callback, best_effort_qos)
#         self.image_sub = self.create_subscription(
#             Image, '/camera_optical/image_raw', self.image_callback, best_effort_qos)
#         self.camera_info_sub = self.create_subscription(
#             CameraInfo, '/camera_optical/camera_info', self.camera_info_callback, best_effort_qos)
#         self.scan_sub = self.create_subscription(
#             LaserScan, '/lidar/scan', self.scan_callback, best_effort_qos)
        
#         # Timer to publish synchronized data at fixed rate
#         self.timer = self.create_timer(0.1, self.publish_synchronized)  # 10Hz
        
#         self.get_logger().info("Complete sync node started - publishing at 10Hz")

#     def odom_callback(self, msg):
#         with self.odom_lock:
#             self.latest_odom = msg

#     def scan_callback(self, msg):
#         with self.scan_lock:
#             self.latest_scan = msg

#     def image_callback(self, msg):
#         """Convert YUV422 to BGR8."""
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
#             if len(cv_image.shape) == 2:
#                 bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
#             else:
#                 bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_YUV2BGR_YUYV)
            
#             converted_image = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
#             converted_image.header = msg.header
            
#             with self.image_lock:
#                 self.latest_image = converted_image
                
#         except Exception as e:
#             self.get_logger().error(f"Image conversion failed: {str(e)}")

#     def camera_info_callback(self, msg):
#         with self.camera_info_lock:
#             self.latest_camera_info = msg

#     def publish_synchronized(self):
#         """Publish all data with synchronized timestamps."""
#         sync_time = self.get_clock().now().to_msg()
#         published_count = 0
        
#         # Publish odometry if available
#         with self.odom_lock:
#             if self.latest_odom is not None:
#                 sync_odom = self.latest_odom
#                 sync_odom.header.stamp = sync_time
#                 self.pub_odom.publish(sync_odom)
#                 published_count += 1
        
#         # Publish scan if available
#         with self.scan_lock:
#             if self.latest_scan is not None:
#                 sync_scan = self.latest_scan
#                 sync_scan.header.stamp = sync_time
#                 self.pub_scan.publish(sync_scan)
#                 published_count += 1
        
#         # Publish image and camera info together
#         with self.image_lock:
#             with self.camera_info_lock:
#                 if self.latest_image is not None and self.latest_camera_info is not None:
#                     sync_image = self.latest_image
#                     sync_image.header.stamp = sync_time
#                     self.pub_image.publish(sync_image)
                    
#                     sync_camera_info = self.latest_camera_info
#                     sync_camera_info.header.stamp = sync_time
#                     self.pub_camera_info.publish(sync_camera_info)
                    
#                     published_count += 2
        
#         if published_count >= 3:  # At least odom, scan, and image
#             self.get_logger().info(f"Published {published_count} synchronized messages", 
#                                   throttle_duration_sec=2.0)

# def main(args=None):
#     rclpy.init(args=args)
#     node = CompleteSyncNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()'