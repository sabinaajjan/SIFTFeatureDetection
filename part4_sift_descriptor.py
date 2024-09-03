#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from vision.part1_harris_corner import compute_image_gradients
from torch import nn
from typing import Tuple


"""
Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells. 
"""


def get_magnitudes_and_orientations(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
    """
    magnitudes = []  # placeholder
    orientations = []  # placeholder

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################


    magnitudes = np.sqrt(np.square(Ix) + np.square(Iy))
    
    orientations = np.arctan2(Iy, Ix)

    
    #raise NotImplementedError('`get_magnitudes_and_orientations()` function ' +
      #  'in `part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return magnitudes, orientations


def get_gradient_histogram_vec_from_patch(
    window_magnitudes: np.ndarray,
    window_orientations: np.ndarray
) -> np.ndarray:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms.

    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively. 
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be
        added to the feature vector left to right then row by row (reading
        order).

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. A useful function to look at would be np.histogram.

    Args:
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch
        window_orientations: (16,16) array representing gradient orientations of
            the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    bin_edges = np.linspace(-np.pi, np.pi, 9)  # 8 bins, 9 edges
    
    wgh = np.zeros((4*4*8, 1))
    
    for i in range(4):
        for j in range(4):
            cell_magnitudes = window_magnitudes[i*4:(i+1)*4, j*4:(j+1)*4]
            cell_orientations = window_orientations[i*4:(i+1)*4, j*4:(j+1)*4]
            hist, _ = np.histogram(cell_orientations, bins=bin_edges, weights=cell_magnitudes)
            wgh[(i*4+j)*8:(i*4+j+1)*8, 0] = hist
    
    #raise NotImplementedError('`get_gradient_histogram_vec_from_patch` ' +
       # 'function in `part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return wgh


def get_feat_vec(
    c: float,
    r: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
    
    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.
    
    Args:
        c: a float, the column (x-coordinate) of the interest point
        r: A float, the row (y-coordinate) of the interest point
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    half_width = feature_width // 2


    r, c = int(r - half_width + 1), int(c - half_width + 1)
    
    fv = get_gradient_histogram_vec_from_patch(magnitudes[r:r+feature_width, c:c+feature_width], orientations[r:r+feature_width, c:c+feature_width])
    
    norm = np.linalg.norm(fv)
    if norm > 0:
        fv /= norm

    fv = np.sqrt(fv)
    
    return fv
    #raise NotImplementedError('`get_feat_vec` function in ' +
        #'`student_sift.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 7.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    # Compute gradients
    # Compute gradients
    Ix, Iy = compute_image_gradients(image_bw)

    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)

    descriptors = []
    for x, y in zip(X, Y):
        fv = get_feat_vec(x, y, magnitudes, orientations, feature_width)
        descriptors.append(fv.flatten())

    return np.array(descriptors)
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    #raise NotImplementedError('`get_SIFT_descriptors` function in ' +
       # '`part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fvs


def rotate_image(
    image: np.ndarray,
    angle: int
) -> np.ndarray :
    """
    Rotate an image by a given angle around its center.

    Args:
    image: numpy array of the image to be rotated
    angle: the angle by which to rotate the image (in degrees)

    Returns:
    Rotated Image as a numpy array

    Note:
    1)Convert the rotation angle from degrees to radians
    2)Find the center of the image (around which the rotation will occur)
    3)Define the rotation matrix for rotating around the image center
    4)Rotation matrix can be [[cos, -sin, center_x*(1-cos)+center_y*sin],
                              [sin,  cos, center_y*(1-cos)-center_x*sin],
                              [0,    0,   1,]]
    5)Apply affine transformation
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    radians = np.deg2rad(angle)
    cos_val, sin_val = np.cos(radians), np.sin(radians)
    
    # Find the center of the image
    center_y, center_x = np.array(image.shape[:2]) / 2 - 0.5
    
    # Compute the inverse rotation matrix
    rotation_matrix_inv = np.array([
        [cos_val, sin_val, center_x - center_x * cos_val - center_y * sin_val],
        [-sin_val, cos_val, center_y + center_x * sin_val - center_y * cos_val]
    ])
    
    # Create an output image filled with zeros
    rotated_image = np.zeros_like(image)
    
    for i in range(rotated_image.shape[0]):
        for j in range(rotated_image.shape[1]):
            # Apply the inverse rotation matrix
            x, y = np.dot(rotation_matrix_inv, np.array([j, i, 1]))
            
            # Interpolate the pixel value
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Get the nearest pixel values around (x, y)
                x0, y0 = int(np.floor(x)), int(np.floor(y))
                x1, y1 = min(x0 + 1, image.shape[1] - 1), min(y0 + 1, image.shape[0] - 1)

                # Calculate the interpolation weights
                a = x - x0
                b = y - y0

                # Interpolate between the four surrounding pixels
                rotated_image[i, j] = (
                    (1 - a) * (1 - b) * image[y0, x0] +
                    a * (1 - b) * image[y0, x1] +
                    (1 - a) * b * image[y1, x0] +
                    a * b * image[y1, x1]
                ).astype(image.dtype)
    
    return rotated_image

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

def crop_center(image, new_width, new_height):
    """
    Crop the central part of an image to the specified dimensions.

    Args:
    image: The image to crop.
    new_width: The target width of the cropped image.
    new_height: The target height of the cropped image.

    Returns:
    cropped image as a numpy array
    """
    height, width = image.shape[:2]
    start_x = width // 2 - new_width // 2
    start_y = height // 2 - new_height // 2
    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]
    return cropped_image

def get_correlation_coeff(
    v1: np.ndarray,
    v2: np.ndarray
) -> float:
    """
    Compute the correlation coefficient between two vectors v1 and v2. Refer to the notebook for the formula.
    Args:
    v1: the first vector
    v2: the second vector
    Returns:
    The scalar correlation coefficient between the two vectors
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                         
    #############################################################################
    # Ensure the vectors are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    dot_product = np.dot(v1, v2)
    
    normalized_v1 = np.linalg.norm(v1)
    normalized_v2 = np.linalg.norm(v2)
    correlation_coefficient = dot_product / (normalized_v1 * normalized_v2)
    
    return correlation_coefficient
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
def get_intensity_based_matches(
    image1: np.ndarray,
    image2: np.ndarray, 
    window_size=64,
    stride=128
) -> np.ndarray:
    """
    Compute intensity-based matches between image1 and image2. For each patch in image1, obtain the patch in image2 with the maximum correlation coefficient.
    Args:
    image1: the first image
    image2: the second image
    window_size: the size of each patch(window) in the images
    stride: the number of pixels by which each patch is shifted to obtain the next patch
    Returns:
    A 3-D numpy array of the form: [[x1, y1],[x2,y2]], where
    x1: x-coordinate of top-left corner of patch in image1
    y1: y-coordinate of top-left corner of patch in image1
    x2: x-coordinate of top-left corner of matching patch in image2
    y2: y-coordinate of top-left corner of matching patch in image2
    """

    # reshaping images to the same dimensions
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])
    image1 = image1[:min_height, :min_width,:]
    image2 = image2[:min_height, :min_width,:]

    #normalizing images for pixel values to be between 0 and 1
    image1 = (image1-np.min(image1))/(np.max(image1)-np.min(image1)) 
    image2 = (image2-np.min(image2))/(np.max(image2)-np.min(image2)) 
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                         
    #############################################################################
    matches = []
    val1 = image1.shape[0] - window_size
    val2 = image1.shape[1] - window_size
    for image1_y in range(0, val1, stride):
        for image2_x in range(0, val2, stride):
            first_patch = image1[image1_y:image1_y + window_size, image2_x:image2_x + window_size].flatten()
            #print(first_patch)
            best_correlation_coefficient = -1
            best_match = (0, 0)
            val3 = image2.shape[0] - window_size
            val4 = image2.shape[1] - window_size
            for y2 in range(0, val3, stride):
                for x2 in range(0, val4, stride):
                    patch2 = image2[y2:y2 + window_size, x2:x2 + window_size].flatten()
                    curr_corr_coefficient = get_correlation_coeff(first_patch, patch2)
                    #print(curr_corr_coefficient)

                    if curr_corr_coefficient > best_correlation_coefficient:
                        best_correlation_coefficient = curr_corr_coefficient
                        best_match = (x2, y2)

            matches.append([[image2_x, image1_y], list(best_match)])

    return np.array(matches)
#raise NotImplementedError('`get_intensity_based_matches` function in ' +
       # '`part4_sift_descriptor.py` needs to be implemented')
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

