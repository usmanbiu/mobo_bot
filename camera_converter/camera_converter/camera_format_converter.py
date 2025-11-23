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