import numpy as np
import cv2
from scipy.spatial.distance import cdist

class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75, distance_metric='euclidean'):
        """
        Initialize feature matcher.
        
        Args:
            ratio_threshold (float): Threshold for Lowe's ratio test
            distance_metric (str): Distance metric for descriptor matching
        """
        self.ratio_threshold = ratio_threshold
        self.distance_metric = distance_metric
    
    def match_descriptors(self, desc1, desc2):
        """
        Match descriptors using Lowe's ratio test.
        
        Args:
            desc1 (numpy.ndarray): First set of descriptors
            desc2 (numpy.ndarray): Second set of descriptors
            
        Returns:
            list: List of DMatch objects
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Compute distance matrix
        distances = cdist(desc1.astype(np.float32), desc2.astype(np.float32), metric=self.distance_metric)
        
        # Find matches using Lowe's ratio test
        matches = []
        for i in range(distances.shape[0]):
            sorted_indices = np.argsort(distances[i])
            if len(sorted_indices) < 2:
                continue
            best_idx = sorted_indices[0]
            second_idx = sorted_indices[1]
            best_dist = distances[i, best_idx]
            second_dist = distances[i, second_idx]
            
            # Ratio test: keep match only if best is significantly better than second
            if second_dist > 0 and best_dist / second_dist < self.ratio_threshold:
                matches.append(cv2.DMatch(i, int(best_idx), float(best_dist)))
        
        return matches

class RANSAC:
    def __init__(self, n_iterations=1000, inlier_threshold=3.0, min_inliers=10):
        """
        Initialize RANSAC algorithm for homography estimation.
        
        Args:
            n_iterations (int): Number of RANSAC iterations
            inlier_threshold (float): Threshold for inlier identification
            min_inliers (int): Minimum number of inliers for a valid model
        """
        self.n_iterations = n_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
    
    def estimate_homography(self, src_points, dst_points):
        """
        Estimate homography matrix using RANSAC.
        
        Args:
            src_points (numpy.ndarray): Source points (N, 2)
            dst_points (numpy.ndarray): Destination points (N, 2)
            
        Returns:
            tuple: (H, inliers) where H is the homography matrix and
                  inliers is a binary mask of inlier matches
        """
        assert src_points.shape[0] == dst_points.shape[0], "Number of points must match"
        assert src_points.shape[0] >= 4, "At least 4 point pairs are required"
        
        # TODO: Implement RANSAC algorithm for homography estimation
        # HINT: 1. Randomly select 4 point pairs
        #       2. Compute homography
        #       3. Transform all points
        #       4. Identify inliers
        #       5. Keep the best model
        
        n_points = src_points.shape[0]
        best_H = None
        best_inliers = np.zeros(n_points, dtype=bool)
        best_inlier_count = 0
        
        for _ in range(self.n_iterations):
            # 1. Randomly select 4 point pairs
            indices = np.random.choice(n_points, 4, replace=False)
            src_subset = src_points[indices]
            dst_subset = dst_points[indices]
            
            # 2. Compute homography from the 4-point sample
            H, status = cv2.findHomography(src_subset, dst_subset, 0)
            if H is None:
                continue
            
            # 3. Transform all source points
            src_h = np.hstack([src_points, np.ones((n_points, 1))])
            projected = (H @ src_h.T).T
            w = projected[:, 2:3]
            w[w == 0] = 1e-10
            projected = projected[:, :2] / w
            
            # 4. Identify inliers by reprojection error
            errors = np.linalg.norm(projected - dst_points, axis=1)
            inliers = errors < self.inlier_threshold
            inlier_count = np.sum(inliers)
            
            # 5. Keep the best model
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                # Refit H using all inliers for a better estimate
                if inlier_count >= 4:
                    best_H, _ = cv2.findHomography(src_points[inliers], dst_points[inliers], 0)
                else:
                    best_H = H
        
        return best_H, best_inliers
    
    def compute_match_quality(self, H, src_points, dst_points, inliers):
        """
        Compute match quality score based on homography transformation.
        
        Args:
            H (numpy.ndarray): Homography matrix
            src_points (numpy.ndarray): Source points
            dst_points (numpy.ndarray): Destination points
            inliers (numpy.ndarray): Binary mask of inlier matches
            
        Returns:
            float: Match quality score
        """
        if H is None or inliers is None:
            return 0.0
        
        n_total = len(inliers)
        n_inliers = int(np.sum(inliers))
        
        # Inlier ratio
        inlier_ratio = n_inliers / n_total if n_total > 0 else 0.0
        
        # Mean reprojection error of inliers
        if n_inliers > 0:
            src_h = np.hstack([src_points[inliers], np.ones((n_inliers, 1))])
            projected = (H @ src_h.T).T
            w = projected[:, 2:3]
            w[w == 0] = 1e-10
            projected = projected[:, :2] / w
            mean_error = np.mean(np.linalg.norm(projected - dst_points[inliers], axis=1))
        else:
            mean_error = float('inf')
        
        # Quality combines inlier ratio and inverse error
        quality_score = inlier_ratio * (1.0 / (1.0 + mean_error))
        
        return quality_score