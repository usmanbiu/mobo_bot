#!/usr/bin/env python3
import numpy as np
from sklearn.cluster import DBSCAN
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
import struct
from std_msgs.msg import Header

class ReflectivityProcessor:
    """Processes RPLIDAR C1 data to detect reflectivity landmarks."""
    
    def __init__(self, reflectivity_threshold=0.5, dbscan_eps=0.3, min_samples=2):
        self.reflectivity_threshold = reflectivity_threshold
        self.dbscan_eps = dbscan_eps
        self.min_samples = min_samples
        
    def process_scan(self, ranges, intensities, angle_min, angle_max, range_min, range_max):
        """
        Process laser scan to extract reflectivity landmarks.
        
        Args:
            ranges: List of range measurements
            intensities: List of intensity/reflectivity measurements
            angle_min: Minimum scan angle
            angle_max: Maximum scan angle
            range_min: Minimum valid range
            range_max: Maximum valid range
            
        Returns:
            landmarks: List of [x, y, z, intensity] for each landmark
            debug_info: Dictionary with processing statistics
        """
        # Input validation
        if len(ranges) != len(intensities):
            raise ValueError("Ranges and intensities must have same length")
        
        # Convert to numpy arrays
        ranges = np.array(ranges)
        intensities = np.array(intensities)
        
        # Filter valid ranges
        valid_mask = (ranges > range_min) & (ranges < range_max)
        valid_ranges = ranges[valid_mask]
        valid_intensities = intensities[valid_mask]
        valid_angles = np.linspace(angle_min, angle_max, len(ranges))[valid_mask]
        
        if len(valid_ranges) == 0:
            return [], {"valid_points": 0, "high_reflectivity": 0, "clusters": 0}
        
        # Convert polar to Cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        z = np.zeros_like(x)  # 2D lidar
        
        # Stack points with intensities
        points = np.column_stack((x, y, z, valid_intensities))
        
        # Find high reflectivity points
        if np.max(valid_intensities) > 0:
            high_reflectivity_mask = valid_intensities > (self.reflectivity_threshold * np.max(valid_intensities))
        else:
            high_reflectivity_mask = valid_intensities > 0.1
            
        high_reflectivity_points = points[high_reflectivity_mask]
        
        debug_info = {
            "valid_points": len(valid_ranges),
            "high_reflectivity": len(high_reflectivity_points),
            "clusters": 0
        }
        
        # Cluster high reflectivity points
        landmarks = []
        if len(high_reflectivity_points) >= self.min_samples:
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.min_samples)
            cluster_labels = clustering.fit_predict(high_reflectivity_points[:, :2])
            
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:  # Ignore noise
                    cluster_points = high_reflectivity_points[cluster_labels == cluster_id]
                    landmark = cluster_points.mean(axis=0)  # Average position and intensity
                    landmarks.append(landmark)
            
            debug_info["clusters"] = len(landmarks)
        
        return landmarks, debug_info
    
    def landmarks_to_pointcloud2(self, landmarks, header):
        """Convert landmarks to PointCloud2 message."""
        if not landmarks:
            return None
            
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        
        # Define fields
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * len(landmarks)
        cloud_msg.is_dense = True
        cloud_msg.height = 1
        cloud_msg.width = len(landmarks)
        
        # Pack data
        cloud_data = bytearray()
        for landmark in landmarks:
            cloud_data.extend(struct.pack('ffff', landmark[0], landmark[1], landmark[2], landmark[3]))
        
        cloud_msg.data = bytes(cloud_data)
        return cloud_msg