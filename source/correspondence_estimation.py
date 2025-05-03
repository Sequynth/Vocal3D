from typing import List, Tuple

import bruteForceCorrespondence
import Correspondences
import numpy as np
import RHC
import torch


class CorrespondenceEstimator:
    def __init__(self, consensus_size: int, iterations: int, min_depth: float, max_depth: float):
        self._correspondences: Tuple[List[np.array], List[np.array]] = None
        self._iterations: int = iterations
        self._consensus_size: int = consensus_size
        self._min_depth: float = min_depth
        self._max_depth: float = max_depth

    def correspondences(self) -> Tuple[List[np.array], List[np.array]]:
        return self._correspondences

    def compute_correspondences(camera, laser, point_image: np.array) -> Tuple[List[np.array], List[np.array]]:
        # return labels in the form of points2d
        return None
    

class RHCEstimator(CorrespondenceEstimator):
    def __init__(self, consensus_size: int, iterations: int, min_depth: float, max_depth: float):
        super().__init__(consensus_size, iterations, min_depth, max_depth)

    def compute_correspondences(self, camera, laser, point_image):
        pixelLocations, laserGridIDs = Correspondences.initialize(
                    laser,
                    camera,
                    point_image.detach().cpu().numpy(),
                    self._min_depth,
                    self._max_depth,
                )
        
        self._correspondences = RHC.RHC(
            laserGridIDs,
            pixelLocations,
            point_image.detach().cpu().numpy(),
            camera,
            laser,
            self._consensus_size,
            self._iterations,
        )
        return self._correspondences
    

class BruteForceEstimator(CorrespondenceEstimator):
    def __init__(self, consensus_size: int, iterations: int, min_depth: float, max_depth: float):
        super().__init__(consensus_size, iterations, min_depth, max_depth)

    def compute_correspondences(self, camera, laser, point_image):
        pixelLocations, laserGridIDs = Correspondences.initialize(
                    laser,
                    camera,
                    point_image.detach().cpu().numpy().copy(),
                    self._min_depth,
                    self._max_depth,
                )
        
        self._correspondences = bruteForceCorrespondence.compute(
            laserGridIDs, 
            pixelLocations, 
            point_image.detach().cpu().numpy().copy(), 
            camera, 
            laser, 
            self._consensus_size, 
            self._iterations)
        
        return self._correspondences