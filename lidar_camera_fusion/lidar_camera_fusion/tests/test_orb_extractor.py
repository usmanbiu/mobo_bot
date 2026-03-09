#!/usr/bin/env python3
import unittest
import numpy as np
import cv2
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from lidar_camera_fusion.modules.orb_extractor import ORBExtractor

class TestORBExtractor(unittest.TestCase):
    
    def setUp(self):
        self.extractor = ORBExtractor(n_features=50)
        self.bridge = CvBridge()
        
    def create_test_image(self, width=640, height=480):
        """Create a test image with some features."""
        # Create chessboard pattern for good feature detection
        image = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = 40
        
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    color = (255, 255, 255)  # White
                else:
                    color = (0, 0, 0)  # Black
                    
                image[i:i+square_size, j:j+square_size] = color
                
        return image
    
    def test_feature_extraction(self):
        """Test ORB feature extraction from test image."""
        test_image = self.create_test_image()
        image_msg = self.bridge.cv2_to_imgmsg(test_image, encoding='bgr8')
        
        keypoints, descriptors, debug_image = self.extractor.extract_features(image_msg)
        
        self.assertGreater(len(keypoints), 0)
        self.assertIsNotNone(descriptors)
        self.assertEqual(descriptors.shape[0], len(keypoints))
        
    def test_empty_image(self):
        """Test feature extraction from empty/blank image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_msg = self.bridge.cv2_to_imgmsg(empty_image, encoding='bgr8')
        
        keypoints, descriptors, debug_image = self.extractor.extract_features(image_msg)
        
        # Should handle empty image gracefully
        self.assertEqual(len(keypoints), 0)
        
    def test_keypoint_message_conversion(self):
        """Test conversion of features to KeyPoint message."""
        test_image = self.create_test_image(320, 240)
        image_msg = self.bridge.cv2_to_imgmsg(test_image, encoding='bgr8')
        
        keypoints, descriptors, _ = self.extractor.extract_features(image_msg)
        
        from std_msgs.msg import Header
        header = Header()
        header.frame_id = "camera_frame"
        
        keypoint_msg = self.extractor._create_keypoint_message(keypoints, descriptors, header)
        
        self.assertEqual(len(keypoint_msg.pts), len(keypoints))
        self.assertEqual(keypoint_msg.header.frame_id, "camera_frame")
        self.assertGreater(len(keypoint_msg.descriptors), 0)

def main():
 # Initialize ROS 2 for testing
    rclpy.init()
    
    # Create test suite explicitly
    suite = unittest.TestLoader().loadTestsFromTestCase(TestORBExtractor)
    
    # Run tests with higher verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Shutdown ROS 2
    rclpy.shutdown()
    
    # Exit with proper code
    exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()