

import numpy as np
from scipy.spatial.transform import Rotation

def get_correct_transform_from_tf():
    """
    Create transformation matrix from the tf output.
    Translation: [-0.000, -0.015, -0.060] in camera_optical frame
    Rotation quaternion: [0.500, -0.500, 0.500, 0.500] (xyzw)
    """
    # Create transformation matrix directly from the tf output
    T = np.array([
        [ 0.000, -1.000,  0.000, -0.000],
        [ 0.000,  0.000, -1.000, -0.015],
        [ 1.000,  0.000,  0.000, -0.060],
        [ 0.000,  0.000,  0.000,  1.000]
    ])
    
    return T

def get_transform_from_quaternion():
    """
    Alternative: Build matrix from quaternion and translation.
    """
    # Translation from tf output
    translation = np.array([-0.000, -0.015, -0.060])
    
    # Quaternion from tf output (xyzw format)
    quaternion_xyzw = np.array([0.500, -0.500, 0.500, 0.500])
    
    # Convert to scipy format (wxyz)
    quaternion_wxyz = np.array([quaternion_xyzw[3], quaternion_xyzw[0], 
                                 quaternion_xyzw[1], quaternion_xyzw[2]])
    
    # Create rotation matrix
    rotation = Rotation.from_quat(quaternion_wxyz)
    R = rotation.as_matrix()
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    
    return T

def test_tf_based_projection():
    """
    Test projection using the actual tf transform.
    """
    print("="*70)
    print("TF-BASED PROJECTION - Using actual camera_optical transform")
    print("Translation: [-0.000, -0.015, -0.060]")
    print("Rotation quaternion: [0.500, -0.500, 0.500, 0.500]")
    print("="*70)
    
    # Camera parameters
    fx, fy = 741.30121, 741.32092
    cx, cy = 292.99457, 204.10432
    
    # Get transform from tf
    T = get_correct_transform_from_tf()
    
    print(f"\nTransformation matrix T_lidar_to_camera_optical:")
    print(f"[{T[0,0]:.6f}, {T[0,1]:.6f}, {T[0,2]:.6f}, {T[0,3]:.6f}]")
    print(f"[{T[1,0]:.6f}, {T[1,1]:.6f}, {T[1,2]:.6f}, {T[1,3]:.6f}]")
    print(f"[{T[2,0]:.6f}, {T[2,1]:.6f}, {T[2,2]:.6f}, {T[2,3]:.6f}]")
    
    # Verify this matches the quaternion version
    T_quat = get_transform_from_quaternion()
    if np.allclose(T, T_quat, atol=1e-10):
        print("\n✓ Matrix matches quaternion conversion")
    else:
        print("\n⚠ Matrix doesn't match quaternion conversion!")
        print("Difference:")
        print(T - T_quat)
    
    # Test points in LiDAR frame
    test_points = [
        [1.0, 0.0, 0.0],      # 1m forward of LiDAR
        [2.0, -0.5, 0.0],     # 2m forward, 0.5m right
        [2.0, 0.5, 0.0],      # 2m forward, 0.5m left
        [0.5, 0.0, 0.0],      # 0.5m forward
        [1.0, 0.0, 0.5],      # 1m forward, 0.5m up
        [1.0, 0.0, -0.5],     # 1m forward, 0.5m down
        [0.1, 0.0, 0.0],      # 100mm forward (close!)
        [0.061, 0.0, 0.0],    # 61mm forward (just in front of camera)
        [0.059, 0.0, 0.0],    # 59mm forward (behind camera!)
    ]
    
    print(f"\n" + "-"*70)
    print("Projection results:")
    print("-"*70)
    
    for i, pt_lidar in enumerate(test_points):
        # Convert to homogeneous
        pt_lidar_h = np.append(pt_lidar, 1.0)
        
        # Transform to camera_optical frame
        pt_cam_h = T @ pt_lidar_h
        Xc, Yc, Zc = pt_cam_h[:3]
        
        print(f"\nPoint {i}:")
        print(f"  LiDAR: X={pt_lidar[0]:.3f}m, Y={pt_lidar[1]:.3f}m, Z={pt_lidar[2]:.3f}m")
        print(f"  Camera optical: X={Xc:.3f}m (right), Y={Yc:.3f}m (down), Z={Zc:.3f}m (forward)")
        
        # Check if in front of camera
        if Zc <= 0.01:  # 1cm threshold
            if Zc < 0:
                print(f"  ❌ BEHIND CAMERA (Z={Zc:.3f}m)")
            else:
                print(f"  ⚠ TOO CLOSE to camera (Z={Zc:.3f}m)")
            continue
        
        # Project to pixels
        u = fx * (Xc / Zc) + cx
        v = fy * (Yc / Zc) + cy
        
        # Normalized coordinates (for debugging)
        u_norm = Xc / Zc
        v_norm = Yc / Zc
        
        print(f"  Normalized: u'={u_norm:.3f}, v'={v_norm:.3f}")
        print(f"  Pixel: u={int(u)}, v={int(v)}")
        
        # Check bounds (640x480)
        if 0 <= u < 640 and 0 <= v < 480:
            print(f"  ✅ IN IMAGE (640x480)")
        else:
            print(f"  ❌ OUT OF BOUNDS")
            
            # Show how far out of bounds
            if u < 0: print(f"     u={u} is {abs(u)} pixels left of image")
            elif u >= 640: print(f"     u={u} is {u-639} pixels right of image")
            if v < 0: print(f"     v={v} is {abs(v)} pixels above image")
            elif v >= 480: print(f"     v={v} is {v-479} pixels below image")

def project_lidar_to_camera(lidar_points, camera_matrix, T_lidar_to_cam=None):
    """
    Project LiDAR points to camera image using the tf transform.
    
    Args:
        lidar_points: Nx3 array in LiDAR frame (X forward, Y left, Z up)
        camera_matrix: 3x3 camera intrinsic matrix
        T_lidar_to_cam: 4x4 transformation matrix (optional, will use default if None)
    
    Returns:
        pixel_coords: Nx2 pixel coordinates (-1 for invalid points)
        valid_mask: Boolean array of valid projections
    """
    if T_lidar_to_cam is None:
        T_lidar_to_cam = get_correct_transform_from_tf()
    
    # Ensure proper shape
    lidar_points = np.array(lidar_points)
    if lidar_points.ndim == 1:
        lidar_points = lidar_points.reshape(1, -1)
    
    N = len(lidar_points)
    
    # Convert to homogeneous coordinates
    ones = np.ones((N, 1))
    lidar_h = np.hstack([lidar_points, ones])
    
    # Transform to camera frame
    camera_h = (T_lidar_to_cam @ lidar_h.T).T
    camera_points = camera_h[:, :3]
    
    # Check points in front of camera
    valid_z = camera_points[:, 2] > 0.01  # Z > 1cm
    valid_points = camera_points[valid_z]
    
    # Initialize output arrays
    pixel_coords = np.full((N, 2), -1, dtype=int)
    valid_mask = np.zeros(N, dtype=bool)
    
    if len(valid_points) == 0:
        return pixel_coords, valid_mask
    
    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Project valid points
    u = fx * (valid_points[:, 0] / valid_points[:, 2]) + cx
    v = fy * (valid_points[:, 1] / valid_points[:, 2]) + cy
    
    pixel_coords_valid = np.column_stack([u, v]).astype(int)
    
    # Store results
    pixel_coords[valid_z] = pixel_coords_valid
    
    # Check image bounds (640x480)
    in_bounds = (pixel_coords_valid[:, 0] >= 0) & (pixel_coords_valid[:, 0] < 640) & \
                (pixel_coords_valid[:, 1] >= 0) & (pixel_coords_valid[:, 1] < 480)
    
    valid_mask[valid_z] = in_bounds
    
    return pixel_coords, valid_mask

def debug_transform_interpretation():
    """
    Debug what this transform actually means.
    """
    print("\n" + "="*70)
    print("TRANSFORM INTERPRETATION")
    print("="*70)
    
    T = get_correct_transform_from_tf()
    
    # Test basis vectors
    basis_vectors = np.eye(3)
    basis_names = ["X (forward)", "Y (left)", "Z (up)"]
    
    print("\nHow LiDAR basis vectors transform to camera frame:")
    for i, (vec, name) in enumerate(zip(basis_vectors, basis_names)):
        vec_h = np.append(vec, 1.0)
        vec_cam_h = T @ vec_h
        vec_cam = vec_cam_h[:3]
        
        print(f"  LiDAR {name}: [1, 0, 0] -> Camera: [{vec_cam[0]:.3f}, {vec_cam[1]:.3f}, {vec_cam[2]:.3f}]")
    
    # Test origin
    origin_lidar = np.array([0, 0, 0, 1])
    origin_cam = T @ origin_lidar
    print(f"\nLiDAR origin in camera frame:")
    print(f"  [{origin_cam[0]:.3f}, {origin_cam[1]:.3f}, {origin_cam[2]:.3f}]")
    
    # Test camera position in LiDAR frame (inverse transform)
    T_inv = np.linalg.inv(T)
    origin_camera = np.array([0, 0, 0, 1])
    camera_in_lidar = T_inv @ origin_camera
    print(f"\nCamera origin in LiDAR frame:")
    print(f"  [{camera_in_lidar[0]:.3f}, {camera_in_lidar[1]:.3f}, {camera_in_lidar[2]:.3f}]")
    
    print("\nInterpretation:")
    print("1. LiDAR X (forward) -> Camera Z (forward)")
    print("2. LiDAR Y (left) -> Camera -X (right)")
    print("3. LiDAR Z (up) -> Camera -Y (down)")
    print("\nCamera position relative to LiDAR:")
    print(f"  X: {camera_in_lidar[0]:.3f}m (forward/backward)")
    print(f"  Y: {camera_in_lidar[1]:.3f}m (left/right)")
    print(f"  Z: {camera_in_lidar[2]:.3f}m (up/down)")

if __name__ == "__main__":
    test_tf_based_projection()
    debug_transform_interpretation()
    
    # Example usage with real projection
    print("\n" + "="*70)
    print("EXAMPLE USAGE")
    print("="*70)
    
    # Create camera matrix
    K = np.array([
        [741.30121, 0, 292.99457],
        [0, 741.32092, 204.10432],
        [0, 0, 1]
    ])
    
    # Test points
    lidar_points = np.array([
        [1.0, 0.0, 0.0],
        [2.0, -0.5, 0.0],
        [2.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
    ])
    
    pixels, valid = project_lidar_to_camera(lidar_points, K)
    
    print(f"\nProjection results:")
    for i, (pt, pix, is_valid) in enumerate(zip(lidar_points, pixels, valid)):
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"Point {i}: LiDAR {pt} -> Pixel {pix} - {status}")


# def get_correct_transform_no_rotation():
#     """
#     CORRECTED: Camera is 60mm IN FRONT, 15mm BELOW LiDAR.
#     Z_camera = X_lidar - 0.060 (point distance minus camera offset)
#     """
#     # Camera position in LiDAR frame:
#     # X: +0.060m (60mm IN FRONT of LiDAR)
#     # Y: 0.000m (same left/right position)
#     # Z: -0.015m (15mm BELOW LiDAR)
    
#     # Rotation from LiDAR to Camera:
#     # LiDAR: X forward, Y left, Z up
#     # Camera: Z forward, X right, Y down
    
#     # So:
#     # X_camera (right) = -Y_lidar
#     # Y_camera (down) = -Z_lidar
#     # Z_camera (forward) = X_lidar
    
#     # But we need to account for camera position!
#     # If camera is at [0.060, 0.000, -0.015] in LiDAR frame,
#     # then for a point at [X, Y, Z] in LiDAR frame:
#     # Relative position = [X-0.060, Y-0.000, Z-(-0.015)] = [X-0.060, Y, Z+0.015]
    
#     # Convert relative position to camera frame:
#     # X_camera = -(Y - 0.000) = -Y
#     # Y_camera = -(Z + 0.015) = -Z - 0.015
#     # Z_camera = (X - 0.060) = X - 0.060
    
#     # So the transformation matrix is:
#     T = np.array([
#         [0, -1,  0,  0.000],   # X_camera = -Y_lidar + 0.000
#         [0,  0, -1, -0.015],   # Y_camera = -Z_lidar - 0.015
#         [1,  0,  0, -0.060],   # Z_camera = X_lidar - 0.060  # FIXED: -0.060 not +0.060
#         [0,  0,  0,  1.000]
#     ])
    
#     return T

# def test_corrected_z_projection():
#     """
#     Test with CORRECTED Z calculation: Z_camera = X_lidar - 0.060
#     """
#     print("="*70)
#     print("CORRECTED Z PROJECTION - Camera 60mm IN FRONT of LiDAR")
#     print("Z_camera = X_lidar - 0.060 (point minus camera offset)")
#     print("="*70)
    
#     tf_buffer = tf2_ros.Buffer()
#     tf_listener = tf2_ros.TransformListener(tf_buffer)

        
#     transform = tf_buffer.lookup_transform("lidar", "camera_optical",
#                     rclpy.time.Time())
#     print(f"transform: {transform}")

#     # Camera parameters
#     fx, fy = 741.30121, 741.32092
#     cx, cy = 292.99457, 204.10432
    
#     # Get CORRECTED transform
#     T = get_correct_transform_no_rotation()
    
#     print(f"\nCORRECTED Transformation matrix:")
#     print(f"[{T[0,0]:.3f}, {T[0,1]:.3f}, {T[0,2]:.3f}, {T[0,3]:.3f}]")
#     print(f"[{T[1,0]:.3f}, {T[1,1]:.3f}, {T[1,2]:.3f}, {T[1,3]:.3f}]")
#     print(f"[{T[2,0]:.3f}, {T[2,1]:.3f}, {T[2,2]:.3f}, {T[2,3]:.3f}]")
    
#     # Test points
#     test_points = [
#         [1.0, 0.0, 0.0],      # 1m forward of LiDAR
#         [2.0, -0.5, 0.0],     # 2m forward, 0.5m right
#         [2.0, 0.5, 0.0],      # 0.5m forward
#         [1.0, 0.0, 0.5],      # 1m forward, 0.5m up
#         [1.0, 0.0, -0.5],     # 1m forward, 0.5m down
#         [0.05, 0.0, 0.0],     # Very close: 50mm forward
#     ]
    
#     for i, pt_lidar in enumerate(test_points):
#         pt_lidar_h = np.append(pt_lidar, 1.0)
#         pt_cam_h = T @ pt_lidar_h
#         Xc, Yc, Zc = pt_cam_h[:3]
        
#         print(f"\nPoint {i}:")
#         print(f"  LiDAR: X={pt_lidar[0]:.2f}m forward, Y={pt_lidar[1]:.2f}m left, Z={pt_lidar[2]:.2f}m up")
#         print(f"  Camera: X={Xc:.3f}m right, Y={Yc:.3f}m down, Z={Zc:.3f}m forward")
        
#         # Check if in front of camera
#         if Zc <= 0.01:
#             print(f"  ❌ BEHIND or TOO CLOSE to camera (Z={Zc:.3f}m)")
#             if Zc < 0:
#                 print(f"     Point is {abs(Zc):.3f}m BEHIND camera")
#             else:
#                 print(f"     Point is only {Zc:.3f}m in front of camera")
#             continue
        
#         # Project
#         u = fx * (Xc / Zc) + cx
#         v = fy * (Yc / Zc) + cy
        
#         print(f"  Pixel: u={int(u)}, v={int(v)}")
#         print(f"  Normalized: u'={Xc/Zc:.3f}, v'={Yc/Zc:.3f}")
        
#         # Check bounds
#         if 0 <= u < 640 and 0 <= v < 480:
#             print(f"  ✅ IN IMAGE (640x480)")
#         else:
#             print(f"  ❌ OUT OF BOUNDS")

# if __name__ == "__main__":
#     test_corrected_z_projection() 

# # from scipy.spatial.transform import Rotation

# # def ros_to_opencv_transform():
# #     """
# #     Create transformation from ROS frame to OpenCV camera frame.
    
# #     ROS: +X forward, +Y left, +Z up
# #     OpenCV Camera: +Z forward, +X right, +Y down
    
# #     To convert: 
# #     1. ROS X (forward) -> OpenCV Z (forward)
# #     2. ROS Y (left) -> OpenCV X (right) = -ROS Y
# #     3. ROS Z (up) -> OpenCV Y (down) = -ROS Z
# #     """
# #     # Rotation matrix for ROS -> OpenCV camera
# #     # X_ros -> Z_cv
# #     # Y_ros -> -X_cv  
# #     # Z_ros -> -Y_cv
# #     R_ros_to_cv = np.array([
# #         [0, -1,  0],  # X_cv = -Y_ros
# #         [0,  0, -1],  # Y_cv = -Z_ros
# #         [1,  0,  0]   # Z_cv = X_ros
# #     ])
    
# #     return R_ros_to_cv

# # def get_full_transformation():
# #     """
# #     Get complete transformation: LiDAR (ROS) -> Camera (OpenCV)
# #     """
# #     # Step 1: Get ROS frame transform (from tf2_echo)
# #     T_ros_lidar_to_ros_camera = np.array([
# #         [-1.0,  0.0,  0.0, -0.06],
# #         [ 0.0, -1.0,  0.0,  0.00],
# #         [ 0.0,  0.0,  1.0, -0.015],
# #         [ 0.0,  0.0,  0.0,  1.0]
# #     ])
    
# #     # Step 2: ROS to OpenCV camera rotation
# #     R_ros_to_cv = ros_to_opencv_transform()
    
# #     # Step 3: Combine transformations
# #     # First: LiDAR_ros -> Camera_ros (using T_ros_lidar_to_ros_camera)
# #     # Then: Camera_ros -> Camera_cv (using R_ros_to_cv)
    
# #     # Extract rotation and translation from ROS transform
# #     R_ros = T_ros_lidar_to_ros_camera[:3, :3]
# #     t_ros = T_ros_lidar_to_ros_camera[:3, 3]
    
# #     # Combine rotations: R_total = R_ros_to_cv @ R_ros
# #     R_total = R_ros_to_cv @ R_ros
    
# #     # Transform translation: t_total = R_ros_to_cv @ t_ros
# #     t_total = R_ros_to_cv @ t_ros
    
# #     # Build final transformation matrix
# #     T_total = np.eye(4)
# #     T_total[:3, :3] = R_total
# #     T_total[:3, 3] = t_total
    
# #     return T_total

# # def debug_coordinate_transformation():
# #     """
# #     Debug the coordinate transformation step-by-step.
# #     """
# #     print("="*70)
# #     print("COORDINATE TRANSFORMATION DEBUG")
# #     print("="*70)
    
# #     # Test point in LiDAR ROS frame
# #     point_lidar_ros = np.array([1.0, 0.0, 0.0])  # 1m forward
    
# #     print(f"\n1. LiDAR point (ROS frame):")
# #     print(f"   [{point_lidar_ros[0]:.2f}, {point_lidar_ros[1]:.2f}, {point_lidar_ros[2]:.2f}]")
# #     print(f"   X: forward, Y: left, Z: up")
    
# #     # Step 1: Transform to camera ROS frame
# #     T_ros = np.array([
# #         [-1.0,  0.0,  0.0, -0.06],
# #         [ 0.0, -1.0,  0.0,  0.00],
# #         [ 0.0,  0.0,  1.0, -0.015],
# #         [ 0.0,  0.0,  0.0,  1.0]
# #     ])
    
# #     point_ros_h = np.append(point_lidar_ros, 1.0)
# #     point_camera_ros_h = T_ros @ point_ros_h
# #     point_camera_ros = point_camera_ros_h[:3]
    
# #     print(f"\n2. After ROS transform (camera ROS frame):")
# #     print(f"   [{point_camera_ros[0]:.3f}, {point_camera_ros[1]:.3f}, {point_camera_ros[2]:.3f}]")
# #     print(f"   Still in ROS convention: X forward, Y left, Z up")
    
# #     # Step 2: Convert to OpenCV camera frame
# #     R_ros_to_cv = ros_to_opencv_transform()
# #     point_camera_cv = R_ros_to_cv @ point_camera_ros
    
# #     print(f"\n3. After ROS->OpenCV conversion (camera OpenCV frame):")
# #     print(f"   [{point_camera_cv[0]:.3f}, {point_camera_cv[1]:.3f}, {point_camera_cv[2]:.3f}]")
# #     print(f"   OpenCV convention: Z forward, X right, Y down")
    
# #     # Check if in front of camera
# #     if point_camera_cv[2] > 0.01:
# #         print(f"\n✅ POINT IS IN FRONT OF CAMERA")
# #         print(f"   Z_camera = {point_camera_cv[2]:.3f} > 0")
# #     else:
# #         print(f"\n❌ POINT IS BEHIND CAMERA")
# #         print(f"   Z_camera = {point_camera_cv[2]:.3f} <= 0")
    
# #     return point_camera_cv

# # def project_lidar_to_camera_correct(lidar_points_ros):
# #     """
# #     Correct projection from LiDAR (ROS frame) to camera (OpenCV frame).
    
# #     Args:
# #         lidar_points_ros: Nx3 array in ROS frame (X forward, Y left, Z up)
    
# #     Returns:
# #         pixel_coords: Nx2 pixel coordinates
# #         valid_mask: Boolean array of valid projections
# #     """
# #     # Camera intrinsics (OpenCV convention)
# #     fx, fy = 741.30121, 741.32092
# #     cx, cy = 292.99457, 204.10432
# #     K = np.array([[fx, 0, cx],
# #                   [0, fy, cy],
# #                   [0, 0, 1]])
    
# #     # Get full transformation matrix
# #     T_total = get_full_transformation()
    
# #     # Convert to homogeneous coordinates
# #     lidar_points_ros = np.array(lidar_points_ros)
# #     if lidar_points_ros.ndim == 1:
# #         lidar_points_ros = lidar_points_ros.reshape(1, -1)
    
# #     ones = np.ones((lidar_points_ros.shape[0], 1))
# #     lidar_homogeneous = np.hstack([lidar_points_ros, ones])
    
# #     # Transform to OpenCV camera frame
# #     points_camera_homogeneous = (T_total @ lidar_homogeneous.T).T
# #     points_camera = points_camera_homogeneous[:, :3]
    
# #     print(f"\nPoints in OpenCV camera frame (first 3):")
# #     for i in range(min(3, len(points_camera))):
# #         print(f"  [{points_camera[i, 0]:.3f}, {points_camera[i, 1]:.3f}, {points_camera[i, 2]:.3f}]")
    
# #     # Filter points in front of camera (Z > 0 in OpenCV frame)
# #     valid_z = points_camera[:, 2] > 0.01
# #     points_valid = points_camera[valid_z]
    
# #     if len(points_valid) == 0:
# #         return np.full((len(lidar_points_ros), 2), -1), np.zeros(len(lidar_points_ros), dtype=bool)
    
# #     # Project to pixel coordinates
# #     points_normalized = points_valid[:, :2] / points_valid[:, 2:3]
# #     ones_norm = np.ones((len(points_normalized), 1))
# #     points_homogeneous_2d = np.hstack([points_normalized, ones_norm])
# #     pixel_coords_homogeneous = (K @ points_homogeneous_2d.T).T
# #     pixel_coords = pixel_coords_homogeneous[:, :2].astype(int)
    
# #     # Create full output
# #     pixel_coords_full = np.full((len(lidar_points_ros), 2), -1, dtype=int)
# #     pixel_coords_full[valid_z] = pixel_coords
    
# #     # Check image bounds
# #     width, height = 640, 480
# #     valid_mask = np.zeros(len(lidar_points_ros), dtype=bool)
    
# #     if len(pixel_coords) > 0:
# #         in_bounds = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) & \
# #                     (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
# #         valid_mask[valid_z] = in_bounds
    
# #     return pixel_coords_full, valid_mask

# # # Test with your points
# # def test_correct_projection():
# #     """Test the corrected projection."""
    
# #     # Your test points in ROS frame (X forward, Y left, Z up)
# #     test_points_ros = np.array([
# #         [1.0, 0.0, 0.0],    # 1m forward, on ground
# #         [2.0, -0.5, 0.0],   # 2m forward, 0.5m right (negative Y = right in ROS)
# #         [0.5, 0.0, 0.5],    # 0.5m forward, 0.5m up
# #         [0.3, -0.2, 0.1],   # Close right, slightly up
# #         [0.3, 0.2, 0.1],    # Close left, slightly up
# #     ])
    
# #     print("Testing corrected projection...")
# #     print("="*70)
    
# #     # Debug coordinate transformation for first point
# #     debug_coordinate_transformation()
    
# #     print("\n" + "="*70)
# #     print("PROJECTING ALL TEST POINTS")
# #     print("="*70)
    
# #     # Project all points
# #     pixel_coords, valid_mask = project_lidar_to_camera_correct(test_points_ros)
    
# #     for i in range(len(test_points_ros)):
# #         print(f"\nPoint {i}:")
# #         print(f"  LiDAR (ROS): [{test_points_ros[i, 0]:.2f}, {test_points_ros[i, 1]:.2f}, {test_points_ros[i, 2]:.2f}]")
# #         print(f"  Pixel: u={pixel_coords[i, 0]}, v={pixel_coords[i, 1]}")
# #         print(f"  Valid: {valid_mask[i]}")
# #         if valid_mask[i]:
# #             print(f"  ✅ In image (640x480)")
# #         else:
# #             print(f"  ❌ Not in image")

# # # Alternative: All-in-one function
# # def transform_and_project_simple(lidar_point_ros):
# #     """
# #     Simplified all-in-one transformation and projection.
    
# #     Args:
# #         lidar_point_ros: [X, Y, Z] in ROS frame (X forward, Y left, Z up)
    
# #     Returns:
# #         u, v pixel coordinates (or None if invalid)
# #     """
# #     # Step 1: Apply ROS transform (from tf2_echo)
# #     X_ros = lidar_point_ros[0]
# #     Y_ros = lidar_point_ros[1] 
# #     Z_ros = lidar_point_ros[2]
    
# #     # Transform to camera ROS frame (180° rotation around Z)
# #     X_cam_ros = -X_ros - 0.06      # X in camera ROS frame
# #     Y_cam_ros = -Y_ros + 0.00      # Y in camera ROS frame  
# #     Z_cam_ros = Z_ros - 0.015      # Z in camera ROS frame
    
# #     # Step 2: Convert ROS to OpenCV camera frame
# #     # ROS: X forward, Y left, Z up
# #     # OpenCV: Z forward, X right, Y down
# #     X_cv = -Y_cam_ros              # ROS Y (left) -> OpenCV X (right) = -ROS Y
# #     Y_cv = -Z_cam_ros              # ROS Z (up) -> OpenCV Y (down) = -ROS Z
# #     Z_cv = X_cam_ros               # ROS X (forward) -> OpenCV Z (forward)
    
# #     # Check if in front of camera
# #     if Z_cv <= 0.01:
# #         return None
    
# #     # Step 3: Project to pixels
# #     fx, fy = 741.30121, 741.32092
# #     cx, cy = 292.99457, 204.10432
    
# #     u = int(fx * (X_cv / Z_cv) + cx)
# #     v = int(fy * (Y_cv / Z_cv) + cy)
    
# #     return u, v

# # # Run the test
# # if __name__ == "__main__":
# #     test_correct_projection()


# # # import numpy as np


# # # def debug_projection_issue():
# # #     """
# # #     Run a diagnostic test with known values.
# # #     """
# # #     print("="*70)
# # #     print("PROJECTION DEBUG DIAGNOSTIC")
# # #     print("="*70)
    
# # #     # Your camera parameters
# # #     fx, fy = 741.30121, 741.32092
# # #     cx, cy = 292.99457, 204.10432
    
# # #     # Create camera matrix
# # #     K = np.array([
# # #         [fx, 0, cx],
# # #         [0, fy, cy],
# # #         [0, 0, 1]
# # #     ])
    
# # #     print(f"\nCamera matrix K:")
# # #     print(K)
    
# # #     # Transformation matrix from tf2_echo output
# # #     T = np.array([
# # #         [-1.0,  0.0,  0.0, -0.06],
# # #         [ 0.0, -1.0,  0.0,  0.00],
# # #         [ 0.0,  0.0,  1.0,  0.015],  # Changed from -0.015 to +0.015
# # #         [ 0.0,  0.0,  0.0,  1.0]
# # #     ])
    
# # #     print(f"\nTransformation matrix T (from tf2_echo):")
# # #     print(T)
    
# # #     # Test with your example points
# # #     test_points = np.array([
# # #     [1.0, 0.0, 0.5],    # 1m forward, 0.5m up
# # #     [2.0, -0.5, 0.3],   # 2m forward, 0.5m right, 0.3m up
# # #     [0.5, 0.0, 0.2],    # 0.5m forward, 0.2m up
# # #     [0.3, -0.2, 0.1],   # Close right, 0.1m up
# # #     [0.3, 0.2, 0.1],    # Close left, 0.1m up
# # #     ])
    
# # #     print(f"\nTest points (Lidar frame, meters):")
# # #     for i, pt in enumerate(test_points):
# # #         print(f"  Point {i}: [{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]")
    
# # #     # Step-by-step projection
# # #     print(f"\n" + "-"*50)
# # #     print("STEP-BY-STEP PROJECTION:")
# # #     print("-"*50)
    
# # #     for i, pt_lidar in enumerate(test_points):
# # #         print(f"\nPoint {i}: [{pt_lidar[0]:.2f}, {pt_lidar[1]:.2f}, {pt_lidar[2]:.2f}]")
        
# # #         # Step 1: Convert to homogeneous
# # #         pt_lidar_h = np.append(pt_lidar, 1.0)
# # #         print(f"  1. Homogeneous (lidar): {pt_lidar_h}")
        
# # #         # Step 2: Transform to camera frame
# # #         pt_cam_h = T @ pt_lidar_h
# # #         print(f"  2. In camera frame: {pt_cam_h[:3]}")
        
# # #         # Step 3: Check if in front
# # #         if pt_cam_h[2] <= 0.01:
# # #             print(f"  3. ❌ BEHIND CAMERA (Z={pt_cam_h[2]:.3f})")
# # #             continue
        
# # #         # Step 4: Perspective division
# # #         u_norm = pt_cam_h[0] / pt_cam_h[2]
# # #         v_norm = pt_cam_h[1] / pt_cam_h[2]
# # #         print(f"  3. Normalized: u'={u_norm:.3f}, v'={v_norm:.3f}")
        
# # #         # Step 5: Apply camera matrix
# # #         u = fx * u_norm + cx
# # #         v = fy * v_norm + cy
# # #         print(f"  4. Pixel coordinates: u={u:.1f}, v={v:.1f}")
        
# # #         # Step 6: Check bounds
# # #         if 0 <= u < 640 and 0 <= v < 480:
# # #             print(f"  5. ✅ IN IMAGE (640x480)")
# # #         else:
# # #             print(f"  5. ❌ OUT OF BOUNDS")

# # # def main():
# # #     debug_projection_issue()

# # # if __name__ == '__main__':
# # #     main()
