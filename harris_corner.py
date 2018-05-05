# GROUPWORK - HARRIS CORNER

# MADLEINA CADUFF

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage import io
from scipy.ndimage import filters
from time import time
import os


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
    best_row = 0
    best_col = 0

    # apply a first derivative Gaussian filter to both axes
    dx = filters.gaussian_filter1d(img, axis=0, sigma=sigma, order=1)
    dy = filters.gaussian_filter1d(img, axis=1, sigma=sigma, order=1)

    # compute product of derivatives at each pixel
    Ix2 = dx**2
    Iy2 = dy**2

    # Loop through image
    for row in range(half_filt, imCols - half_filt):
        for col in range(half_filt, imRows - half_filt):
            # Compute sums of products of derivatives at each pixel
            Sx2 = Ix2[row - half_filt: row + half_filt + 1, col - half_filt: col + half_filt + 1].sum()
            Sy2 = Iy2[row - half_filt: row + half_filt + 1, col - half_filt: col + half_filt + 1].sum()

            # Find determinant and trace, use to get corner response
            det = Sx2 * Sy2
            trace = Sx2 + Sy2
            r = det - k*(trace**2)

            # Only keep the best-scoring corner response
            if r > best_r:
                best_r = r
                best_row = row
                best_col = col

    return best_row, best_col


start = time()

images = io.imread_collection(os.path.join('./project_data/a/', '*.png'))  # read all images from folder a

# parameters for the harris corner
sigma = 2  # sd for Gaussian filter
filter_size = 8  # filter size for calculating the derivative
k = 0.06

half_window_size = 8  # size of frame to extract around the position in the previous image

for j, img in enumerate(images):
    print('Processing image Nr. ' + str(j))
    if j == 0:  # add known starting positions
        col_img_pos = 348
        old_col_pos = 348
        row_img_pos = 191
        old_row_pos = 191
    else:
        image = np.copy(img)
        # extract only a certain frame of the image, based on the position in the previous image
        frame = image[old_col_pos - half_window_size: old_col_pos + half_window_size + 1,
                old_row_pos - half_window_size: old_row_pos + half_window_size + 1]

        frame = color.rgb2gray(frame)  # convert to greyscale
        # get corner position in the frame
        row_frame_pos, col_frame_pos = harris_corner(frame, sigma, filter_size, k)
        # get corner position in full image
        row_img_pos = row_frame_pos + old_row_pos - half_window_size
        col_img_pos = col_frame_pos + old_col_pos - half_window_size
        # update "old" positions
        old_row_pos = row_img_pos
        old_col_pos = col_img_pos

    print('row_pos: ' + str(row_img_pos), '\ncol_pos: ' + str(col_img_pos))
    plt.scatter(col_img_pos, row_img_pos, c='r', marker='o')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

stop = time()
print('Time elapsed: ' + str(stop - start) + ' s')

