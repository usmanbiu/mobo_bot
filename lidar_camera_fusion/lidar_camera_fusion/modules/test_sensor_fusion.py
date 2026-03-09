#!/usr/bin/env python3
import unittest
import numpy as np
import cv2
from geometry_msgs.msg import TransformStamped
from modules.sensor_fusion import SensorFusion

class TestSensorFusion(unittest.TestCase):
    
    def setUp(self):
        self.fusion = SensorFusion(max_association_distance=50.0)
        
        # Create test camera parameters
        self.camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros(5)
        
    def create_test_transform(self):
        """Create a simple identity transform for testing."""
        transform = TransformStamped()
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        return transform
    
    def test_projection(self):
        """Test projection of lidar landmarks to camera coordinates."""
        landmarks = [
            [1.0, 0.0, 0.0, 0.8],  # Point in front of camera
            [0.0, 1.0, 0.0, 0.9],  # Point to the right
        ]
        
        transform = self.create_test_transform()
        
        projected_points = self.fusion.project_lidar_to_camera(
            landmarks, transform, self.camera_matrix, self.dist_coeffs
        )
        
        self.assertEqual(len(projected_points), 2)
        
        # Check that points are projected to reasonable pixel coordinates
        for u, v, intensity, landmark in projected_points:
            self.assertIsInstance(u, (int, np.integer))
            self.assertIsInstance(v, (int, np.integer))
            self.assertGreaterEqual(u, 0)
            self.assertGreaterEqual(v, 0)
            
    def test_feature_association(self):
        """Test association between projected points and ORB features."""
        # Create test projected points
        projected_points = [
            (320, 240, 0.8, [1.0, 0.0, 0.0, 0.8]),  # Center of image
            (100, 100, 0.9, [0.5, 0.5, 0.0, 0.9]),  # Top-left
        ]
        
        # Create test ORB features
        keypoints = [
            cv2.KeyPoint(320, 240, 10),  # Exactly at first projected point
            cv2.KeyPoint(110, 110, 10),  # Close to second point
            cv2.KeyPoint(500, 400, 10),  # Far away (should not associate)
        ]
        
        descriptors = np.random.randint(0, 256, (3, 32), dtype=np.uint8)
        image_shape = (480, 640)
        
        associations, _ = self.fusion.associate_features(
            projected_points, keypoints, descriptors, image_shape
        )
        
        # Should find 2 associations
        self.assertEqual(len(associations), 2)
        
    def test_hybrid_descriptor_creation(self):
        """Test creation of hybrid ORB+reflectivity descriptors."""
        # Mock association data
        associations = [
            ([1.0, 0.0, 0.0, 200],  # landmark with high intensity
             cv2.KeyPoint(100, 100, 10),
             np.random.randint(0, 256, 32, dtype=np.uint8),  # ORB descriptor
             5.0),  # distance
        ]
        
        hybrid_descriptors, stats = self.fusion.create_hybrid_descriptors(associations)
        
        self.assertEqual(stats["successful_fusions"], 1)
        self.assertEqual(len(hybrid_descriptors), 1)
        
        # Hybrid descriptor should be ORB (256 bits = 32 floats) + 2 additional values
        self.assertEqual(len(hybrid_descriptors[0]), 32 + 2)

def main():
    unittest.main()

if __name__ == '__main__':
    main()