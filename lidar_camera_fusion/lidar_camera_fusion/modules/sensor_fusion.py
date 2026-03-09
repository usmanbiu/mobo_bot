#!/usr/bin/env python3
import numpy as np
import cv2
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

class SensorFusion:
    """Fuses lidar landmarks with camera features."""
    
    def __init__(self, max_association_distance=50.0):
        self.max_association_distance = max_association_distance
        
    def project_lidar_to_camera(self, landmarks, transform, camera_matrix, dist_coeffs):
        """
        Project lidar landmarks to camera image coordinates.
        
        Args:
            landmarks: List of [x, y, z, intensity] landmarks
            transform: TF2 transform from lidar to camera frame
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            
        Returns:
            projected_points: List of (u, v, intensity, original_landmark) for each successfully projected point
        """
        projected_points = []
        
        for landmark in landmarks:
            # Create 3D point in lidar frame
            pt_lidar = PointStamped()
            pt_lidar.point.x = landmark[0]
            pt_lidar.point.y = landmark[1]
            pt_lidar.point.z = landmark[2]
            
            try:
                # Transform to camera frame
                pt_camera = do_transform_point(pt_lidar, transform)
                
                # Project to 2D pixel coordinates
                point_3d = np.array([[pt_camera.point.x, pt_camera.point.y, pt_camera.point.z]])
                uv, _ = cv2.projectPoints(point_3d, np.zeros(3), np.zeros(3), 
                                        camera_matrix, dist_coeffs)
                u, v = uv[0][0].astype(int)
                
                projected_points.append((u, v, landmark[3], landmark))
                
            except Exception as e:
                continue
                
        return projected_points
    
    def associate_features(self, projected_points, keypoints, descriptors, image_shape):
        """
        Associate projected lidar points with ORB features.
        
        Args:
            projected_points: List of (u, v, intensity, landmark) from projection
            keypoints: List of cv2.KeyPoint
            descriptors: ORB descriptors
            image_shape: (height, width) of the image
            
        Returns:
            associations: List of (landmark, keypoint, descriptor, distance)
            debug_image: Image with associations visualized
        """
        associations = []
        
        # Create debug image (will be populated by caller)
        debug_image = None
        
        if not keypoints or not projected_points:
            return associations, debug_image
        
        for u, v, intensity, landmark in projected_points:
            # Check if point is within image bounds
            if not (0 <= u < image_shape[1] and 0 <= v < image_shape[0]):
                continue
                
            # Find nearest ORB feature
            min_dist = float('inf')
            best_kp = None
            best_desc = None
            best_idx = -1
            
            for i, kp in enumerate(keypoints):
                kp_u, kp_v = int(kp.pt[0]), int(kp.pt[1])
                dist = np.sqrt((kp_u - u)**2 + (kp_v - v)**2)
                
                if dist < min_dist and dist < self.max_association_distance:
                    min_dist = dist
                    best_kp = kp
                    best_desc = descriptors[i] if descriptors is not None else None
                    best_idx = i
            
            if best_kp is not None:
                associations.append((landmark, best_kp, best_desc, min_dist))
                
        return associations, debug_image
    
    def create_hybrid_descriptors(self, associations):
        """
        Create hybrid descriptors by fusing ORB and reflectivity.
        
        Args:
            associations: List of (landmark, keypoint, descriptor, distance)
            
        Returns:
            hybrid_descriptors: List of fused descriptors
            fusion_stats: Dictionary with fusion statistics
        """
        hybrid_descriptors = []
        fusion_stats = {
            "total_associations": len(associations),
            "successful_fusions": 0
        }
        
        for landmark, keypoint, descriptor, distance in associations:
            if descriptor is not None:
                try:
                    # Convert ORB binary descriptor to float array
                    orb_float = np.unpackbits(descriptor).astype(np.float32)
                    
                    # Normalize reflectivity (0-1 range)
                    reflectivity_norm = np.clip(landmark[3] / 255.0, 0.0, 1.0)
                    
                    # Create hybrid descriptor: ORB + reflectivity
                    hybrid_desc = np.concatenate([orb_float, [reflectivity_norm, distance]])
                    hybrid_descriptors.append(hybrid_desc)
                    
                    fusion_stats["successful_fusions"] += 1
                    
                except Exception as e:
                    continue
        
        return hybrid_descriptors, fusion_stats