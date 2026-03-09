#!/usr/bin/env python3
import unittest
import numpy as np
import rclpy
from lidar_camera_fusion.modules.reflectivity_processor import ReflectivityProcessor

class TestReflectivityProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = ReflectivityProcessor(
            reflectivity_threshold=0.5,
            dbscan_eps=0.3,
            min_samples=2
        )
        
    def test_empty_scan(self):
        """Test processing of empty scan data."""
        ranges = []
        intensities = []
        landmarks, debug_info = self.processor.process_scan(
            ranges, intensities, -3.14, 3.14, 0.1, 10.0
        )
        
        self.assertEqual(len(landmarks), 0)
        self.assertEqual(debug_info["valid_points"], 0)
        
    def test_single_high_reflectivity_point(self):
        """Test detection of single high reflectivity point."""
        ranges = [1.0, 2.0, 1.5]
        intensities = [0.1, 0.9, 0.2]  # One high reflectivity
        
        landmarks, debug_info = self.processor.process_scan(
            ranges, intensities, -1.0, 1.0, 0.1, 10.0
        )
        
        # Should find clusters only if min_samples is met
        self.assertEqual(debug_info["high_reflectivity"], 1)
        print(f"Debug: high_reflectivity = {debug_info['high_reflectivity']}")
        
    def test_cluster_formation(self):
        """Test formation of reflectivity clusters."""
        # Create three close points with high reflectivity
        ranges = [1.0, 1.1, 1.2, 3.0, 1.5, 2, 1.8]
        intensities = [0.8, 0.9, 0.85, 0.1, 0.6, 0.7, 0.5]
        
        landmarks, debug_info = self.processor.process_scan(
            ranges, intensities, -0.5, 0.5, 0.1, 10.0
        )
        
        print(f"Number of clusters found: {debug_info['clusters']}")
        print(f"Debug info: {debug_info}")
        
        # Should form one cluster from the three close points
        self.assertGreaterEqual(debug_info["clusters"], 1)
        self.assertEqual(debug_info["high_reflectivity"], 6)
        
        # CORRECTED: Print the number of clusters

        
        # Additional assertion for clarity
        self.assertEqual(debug_info["clusters"], 1, 
                        f"Expected 1 cluster, got {debug_info['clusters']}")

    def test_pointcloud_conversion(self):
        """Test conversion of landmarks to PointCloud2."""
        landmarks = [
            [1.0, 2.0, 0.0, 0.8],
            [1.5, 2.5, 0.0, 0.9]
        ]
        
        from std_msgs.msg import Header
        header = Header()
        header.frame_id = "test_frame"
        
        cloud_msg = self.processor.landmarks_to_pointcloud2(landmarks, header)
        
        self.assertIsNotNone(cloud_msg)
        self.assertEqual(cloud_msg.width, 2)
        self.assertEqual(cloud_msg.header.frame_id, "test_frame")
        print(f"PointCloud2 created with {cloud_msg.width} points")

def main():
    # Initialize ROS 2 for testing
    rclpy.init()
    
    # Create test suite explicitly
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReflectivityProcessor)
    
    # Run tests with higher verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Shutdown ROS 2
    rclpy.shutdown()
    
    # Exit with proper code
    exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()