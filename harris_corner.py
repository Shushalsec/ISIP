#!/usr/bin/env python3

# GROUPWORK - HARRIS CORNER

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from scipy.ndimage import filters
from time import time
from glob import glob


def harris_corner(img, sigma, filter_size, k):
    """
    Finds and returns list of corners and new image with corners drawn
    :param img: The original image
    :param sigma: The standard deviation for the Gaussian filter
    :param filter_size: The size of the box filter used for computing sum of derivatives
    :param k: Harris corner constant. Usually 0.04 - 0.06
    :return:
    """
    imCols, imRows = np.shape(img)
    half_filt = filter_size//2
    best_r = 0
    best_x = 0
    best_y = 0

    # apply a first derivative Gaussian filter to both axes
    dx = filters.gaussian_filter1d(img, axis=0, sigma=sigma, order=1)
    dy = filters.gaussian_filter1d(img, axis=1, sigma=sigma, order=1)

    # compute product of derivatives at each pixel
    Ix2 = dx**2
    Iy2 = dy**2

    # Loop through image
    for x in range(half_filt, imCols - half_filt):
        for y in range(half_filt, imRows - half_filt):
            # Compute sums of products of derivatives at each pixel
            Sx2 = Ix2[y - half_filt: y + half_filt + 1, x - half_filt: x + half_filt + 1].sum()
            Sy2 = Iy2[y - half_filt: y + half_filt + 1, x - half_filt: x + half_filt + 1].sum()

            # Find determinant and trace, use to get corner response
            det = Sx2 * Sy2
            trace = Sx2 + Sy2
            r = det - k*(trace**2)

            # Only keep the best-scoring corner response
            if r > best_r:
                best_r = r
                best_x = x
                best_y = y

    return best_x, best_y


start = time()

images_names = glob("./project_data/a/*.png")  # read all images from folder a
start_pos = 348, 191  # of pictures a
#images_names = glob("./project_data/b/*.png")  # read all images from folder b
#start_pos = 439, 272  # of pictures b

images_names.sort()

# parameters for the harris corner
sigma = 1  # sd for Gaussian filter
filter_size = 10  # filter size for calculating the derivative
k = 0.06

half_window_size = 20  # size of frame to extract around the position in the previous image

positions = []

for j in range(len(images_names)):
    print('Processing image Nr. ' + str(j))
    img = plt.imread(images_names[j])
    if j == 0:  # add known starting positions
        x_img_pos, y_img_pos = start_pos
    else:
        image = np.copy(img)
        # extract only a certain frame of the image, based on the position in the previous image
        frame = image[positions[j-1][1] - half_window_size: positions[j-1][1] + half_window_size + 1,
                positions[j-1][0] - half_window_size: positions[j-1][0] + half_window_size + 1]

        frame = color.rgb2gray(frame)  # convert to greyscale
        # plt.imshow(frame)
        # plt.show()
        # get corner position in the frame
        x_frame_pos, y_frame_pos = harris_corner(frame, sigma, filter_size, k)
        # get corner position in full image
        x_img_pos = positions[j-1][0] - half_window_size + x_frame_pos + filter_size//2
        y_img_pos = positions[j-1][1] - half_window_size + y_frame_pos + filter_size//2

    positions.append([x_img_pos, y_img_pos])
    print('x_pos: ' + str(x_img_pos), '\ny_pos: ' + str(y_img_pos))
    plt.scatter(x_img_pos, y_img_pos, c='r', marker='o')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

stop = time()
print('Time elapsed: ' + str(stop - start) + ' s')

