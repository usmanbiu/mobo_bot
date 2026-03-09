#!/usr/bin/env python3
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rtabmap_msgs.msg import KeyPoint, GlobalDescriptor
from std_msgs.msg import Header

class ORBExtractor:
    """Correct implementation of ORB feature extraction for RTAB-Map."""
    
    def __init__(self, n_features=100, scale_factor=1.2, n_levels=8, fast_threshold=20):
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            fastThreshold=fast_threshold,
            edgeThreshold=31,
            patchSize=31
        )
        self.bridge = CvBridge()
        
    def extract_features(self, image_msg, max_features=100):
        """
        Extract ORB features from ROS Image message.
        
        Args:
            image_msg: sensor_msgs/Image message
            max_features: Maximum number of features to extract
            
        Returns:
            keypoint_msg: rtabmap_ros/KeyPoint message (batched)
            descriptor_msg: Optional GlobalDescriptor for metadata
            debug_image: Image with keypoints visualized
        """
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            # Convert to grayscale for ORB
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Extract ORB features
            cv_keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
            
            # # Limit number of features if needed
            # if len(cv_keypoints) > max_features:
            #     # Sort by response and keep top features
            #     responses = [kp.response for kp in cv_keypoints]
            #     idxs = np.argsort(responses)[::-1][:max_features]
            #     cv_keypoints = [cv_keypoints[i] for i in idxs]
            #     descriptors = descriptors[idxs]
            
            # Create RTAB-Map KeyPoint message
            keypoint_msg = self._create_keypoint_message(cv_keypoints, descriptors)
            
            # Create GlobalDescriptor for metadata (optional)
            descriptor_msg = self._create_global_descriptor(image_msg.header, len(cv_keypoints))
            
            # Create debug visualization
            debug_image = self._create_debug_image(cv_image, cv_keypoints)
            
            return keypoint_msg, descriptor_msg, debug_image, len(cv_keypoints)
            
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")
    
    def _create_keypoint_message(self, cv_keypoints, descriptors):
        """Convert OpenCV keypoints to RTAB-Map KeyPoint message."""
        keypoint_msg = KeyPoint()
        
        # if not cv_keypoints:
        #     # Return empty but valid message
        #     keypoint_msg.descriptor_size = 32  # ORB descriptor size
        #     keypoint_msg.type = 2  # ORB feature type
        #     return keypoint_msg
        
        # Flatten keypoint positions: [x1, y1, x2, y2, ...]
        # pts_flat = []
        # for kp in cv_keypoints:
        #     pts_flat.extend([int(kp.pt[0]), int(kp.pt[1])])
        # keypoint_msg.pts = pts_flat
        
        # Other keypoint attributes
        keypoint_msg.size = [kp.size for kp in cv_keypoints]
        keypoint_msg.angle = [kp.angle for kp in cv_keypoints]
        keypoint_msg.response = [kp.response for kp in cv_keypoints]
        keypoint_msg.octave = [kp.octave for kp in cv_keypoints]
        keypoint_msg.class_id = [kp.class_id for kp in cv_keypoints]
        
        # Descriptor data (ORB descriptors are binary)
        if descriptors is not None:
            # Ensure descriptors are in correct format (bytes)
            if descriptors.dtype != np.uint8:
                descriptors = descriptors.astype(np.uint8)
            keypoint_msg.descriptors = descriptors.tobytes()
            keypoint_msg.descriptor_size = descriptors.shape[1]  # 32 for ORB
        
        keypoint_msg.type = 2  # 2 = ORB features in RTAB-Map
        
        return keypoint_msg
    
    def _create_global_descriptor(self, image_header, num_features):
        """Create a GlobalDescriptor message for metadata/timing."""
        descriptor_msg = GlobalDescriptor()
        descriptor_msg.header = image_header
        descriptor_msg.type = 1000  # Custom type for feature metadata
        
        # Store metadata as info field (optional)
        import struct
        metadata = struct.pack('I', num_features)  # 4-byte integer
        descriptor_msg.info = metadata
        
        return descriptor_msg
    
    def _create_debug_image(self, cv_image, cv_keypoints):
        """Create debug visualization of keypoints."""
        debug_image = cv_image.copy()
        
        if cv_keypoints:
            # Draw keypoints with size/angle information
            debug_image = cv2.drawKeypoints(
                debug_image,
                cv_keypoints,
                None,
                color=(0, 255, 0),  # Green
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Add feature count text
            cv2.putText(debug_image, f"Features: {len(cv_keypoints)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return debug_image
    
    def verify_keypoint_message(self, keypoint_msg):
        """
        Verify that a KeyPoint message is correctly formatted.
        
        Args:
            keypoint_msg: KeyPoint message to verify
            
        Returns:
            is_valid: True if message is properly formatted
            error_msg: Description of any issues found
        """
        # Check required fields
        if not hasattr(keypoint_msg, 'pts'):
            return False, "Missing pts field"
        
        if not hasattr(keypoint_msg, 'descriptors'):
            return False, "Missing descriptors field"
        
        # Check array lengths are consistent
        if len(keypoint_msg.pts) % 2 != 0:
            return False, f"pts array has odd length ({len(keypoint_msg.pts)})"
        
        num_keypoints = len(keypoint_msg.pts) // 2
        
        # Check other arrays match keypoint count
        arrays_to_check = ['size', 'angle', 'response', 'octave', 'class_id']
        for array_name in arrays_to_check:
            array = getattr(keypoint_msg, array_name, [])
            if len(array) != num_keypoints:
                return False, f"{array_name} has {len(array)} elements, expected {num_keypoints}"
        
        # Check descriptor data
        if keypoint_msg.descriptor_size <= 0:
            return False, f"Invalid descriptor_size: {keypoint_msg.descriptor_size}"
        
        expected_descriptor_bytes = num_keypoints * keypoint_msg.descriptor_size
        actual_descriptor_bytes = len(keypoint_msg.descriptors)
        
        if actual_descriptor_bytes != expected_descriptor_bytes:
            return False, f"Descriptors: {actual_descriptor_bytes} bytes, expected {expected_descriptor_bytes}"
        
        return True, "Message is valid"




# #!/usr/bin/env python3
# import cv2
# import numpy as np
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from rtabmap_msgs.msg import KeyPoint, Point2f

# class ORBExtractor:
#     """Extracts ORB features from images."""
    
#     def __init__(self, n_features=100, scale_factor=1.2, n_levels=8):
#         self.orb = cv2.ORB_create(
#             nfeatures=n_features,
#             scaleFactor=scale_factor,
#             nlevels=n_levels
#         )
#         self.bridge = CvBridge()
        
#     def extract_features(self, image_msg):
#         """
#         Extract ORB features from ROS Image message.
        
#         Args:
#             image_msg: sensor_msgs/Image message
            
#         Returns:
#             keypoints: List of cv2.KeyPoint
#             descriptors: numpy array of ORB descriptors
#             debug_image: Image with keypoints visualized
#         """
#         try:
#             # Convert ROS Image to OpenCV
#             cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
#             # Convert to grayscale for ORB
#             gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
#             # Extract ORB features
#             keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
            
#             # Create debug visualization
#             debug_image = cv_image.copy()
#             cv2.drawKeypoints(debug_image, keypoints, debug_image, 
#                             color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
#             return keypoints, descriptors, debug_image
            
#         except Exception as e:
#             raise ValueError(f"Image processing failed: {str(e)}")
    
#     def features_to_keypoint_msg(self, keypoints, descriptors, header):
#         """Convert ORB features to KeyPoint message."""
#         keypoint_msg = KeyPoint()
#         keypoint_msg.header = header
        
#         if not keypoints:
#             return keypoint_msg
        
#         # Convert keypoints
#         keypoint_msg.pts = [Point2f(x=kp.pt[0], y=kp.pt[1]) for kp in keypoints]
#         keypoint_msg.size = [kp.size for kp in keypoints]
#         keypoint_msg.angle = [kp.angle for kp in keypoints]
#         keypoint_msg.response = [kp.response for kp in keypoints]
#         keypoint_msg.octave = [kp.octave for kp in keypoints]
#         keypoint_msg.class_id = [kp.class_id for kp in keypoints]
        
#         # Convert descriptors
#         if descriptors is not None:
#             keypoint_msg.descriptors = descriptors.tobytes()
#             keypoint_msg.descriptor_size = descriptors.shape[1]
#             keypoint_msg.type = 2  # ORB
            
#         return keypoint_msg