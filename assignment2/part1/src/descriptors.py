import numpy as np
import cv2

class FeatureDescriptor:
    def __init__(self, descriptor_type='SIFT', params=None):
        """
        Initialize feature descriptor.
        
        Args:
            descriptor_type (str): Type of descriptor ('SIFT' or 'SURF')
            params (dict): Parameters for the descriptor
        """
        self.descriptor_type = descriptor_type
        self.params = params if params is not None else {}
        
        self._init_descriptor()
    
    def _init_descriptor(self):
        """
        Initialize the descriptor object based on the type.
        """
        # TODO: Initialize SIFT or SURF descriptor based on self.descriptor_type
        # HINT: Use cv2.SIFT_create() or cv2.xfeatures2d.SURF_create()
        
        if self.descriptor_type == 'SIFT':
            nfeatures = self.params.get('nfeatures', 0)
            nOctaveLayers = self.params.get('nOctaveLayers', 3)
            contrastThreshold = self.params.get('contrastThreshold', 0.04)
            edgeThreshold = self.params.get('edgeThreshold', 10)
            sigma = self.params.get('sigma', 1.6)
            self.descriptor = cv2.SIFT_create(
                nfeatures=nfeatures,
                nOctaveLayers=nOctaveLayers,
                contrastThreshold=contrastThreshold,
                edgeThreshold=edgeThreshold,
                sigma=sigma
            )
            
        elif self.descriptor_type == 'SURF':
            hessianThreshold = self.params.get('hessianThreshold', 400)
            try:
                self.descriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
            except AttributeError:
                raise ValueError("SURF is not available in this OpenCV build. Use SIFT instead.")
            
        else:
            raise ValueError(f"Unsupported descriptor type: {self.descriptor_type}")
    
    def detect_and_compute(self, image, mask=None):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            mask (numpy.ndarray): Optional mask to restrict feature detection
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.descriptor.detectAndCompute(image, mask)
        
        return keypoints, descriptors
    
    def compute_for_keypoints(self, image, keypoints):
        """
        Compute descriptors for given keypoints.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            keypoints (list): List of keypoints
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.descriptor.compute(image, keypoints)
        
        return keypoints, descriptors

class HarrisKeypointExtractor:
    def __init__(self, harris_detector):
        """
        Initialize keypoint extractor based on Harris detector.
        
        Args:
            harris_detector (HarrisDetector): Harris corner detector instance
        """
        self.harris_detector = harris_detector
    
    def detect(self, image, mask=None):
        """
        Detect keypoints using Harris detector.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            mask (numpy.ndarray): Optional mask to restrict feature detection
            
        Returns:
            list: List of cv2.KeyPoint objects
        """
        # Detect Harris corners and convert to cv2.KeyPoint objects
        corners, response = self.harris_detector.detect_corners(image)
        corner_coords = self.harris_detector.get_corner_coordinates(corners)
        
        keypoints = [cv2.KeyPoint(float(x), float(y), 20.0) for x, y in corner_coords]
        
        if mask is not None and len(keypoints) > 0:
            keypoints = [kp for kp in keypoints
                         if mask[int(kp.pt[1]), int(kp.pt[0])] != 0]
        
        return keypoints