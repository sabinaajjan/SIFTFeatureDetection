#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """
    K = X.shape[0]  
    D = feature_width * feature_width  
    fvs = np.zeros((K, D))  
    
    for i, (x, y) in enumerate(zip(X, Y)):
        left = x - (feature_width // 2) if feature_width % 2 == 1 else x - (feature_width // 2) + 1
        right = left + feature_width
        top = y - (feature_width // 2) if feature_width % 2 == 1 else y - (feature_width // 2) + 1
        bottom = top + feature_width
        
        patch = image_bw[top:bottom, left:right]
        
        patch_vector = patch.flatten()
        norm = np.linalg.norm(patch_vector)
        normalized_vector = patch_vector / norm if norm != 0 else patch_vector
        
        fvs[i, :] = normalized_vector
    

    
    #return fvs
    
    

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    #raise NotImplementedError('`compute_normalized_patch_descriptors` ' +
       # 'function in`part2_patch_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
