#!/usr/bin/env python3

# INTRODUCTION TO SIGNAL AND IMAGE PROCESSING 2018 - GROUP PROJECT

# TRACKING SURGICAL INSTRUMENTS USING NCC

# Shushan Toneyan, Madleina Caduff, Judith Bergada Pijuan

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_template


def imgsimplify(img, c, window):
    """
    Returns a patch of the image
    Parameters
    ----------
    img : array_like
        Image that we want to cut or simplify
    c : tuple
        Center of the new image
    window : integer
        Size of the desired final image. It must be an odd number.
    Returns
    -------
    simple_img : array_like
        Image of size window x window that results from cutting the initial img.
    """
    c_x, c_y = c[0], c[1]
    window_half = (window - 1) // 2
    simple_img = img[c_y - window_half:c_y + window_half + 1,
                     c_x - window_half:c_x + window_half + 1]
    return simple_img


if __name__ == '__main__':
    #_______________________________________________________________
    # Activate to work only with set A
    files_names = glob("./project_data/a/*.png")
    c_in = 348, 191
    output = open('output_a.txt', 'w')
    #_______________________________________________________________
    # Activate to work only with set B
    # files_names = glob("./project_data/b/*.png")
    # c_in = 439, 272
    # output = open('output_b.txt', 'w')
    #_______________________________________________________________

    files_names.sort()

    # Parameters
    num_imgs = 10
    window_next = 35
    window_template = window_next // 2

    # Initial conditions
    img = rgb2gray(plt.imread(
        files_names[0]))  # read first image with known starting positions
    previous_imgs = np.zeros((num_imgs, window_template, window_template))
    previous_imgs[0, :, :] = imgsimplify(img, c_in, window_template)

    # Open a .txt file where the coordinates of the points will be writen
    output.write("  image_name\t    x-location\t    y-location\n")
    output.write("{}\t\t{}\t\t{}\n".format(files_names[0], c_in[0], c_in[1]))
    output.flush()

    # Find the desired pixel in all frames
    for i in range(1, len(files_names)):
        # Take next image of the list and its potential regions to find the point
        img_to_plot = plt.imread(files_names[i])
        img_next = rgb2gray(img_to_plot)  # convert to grayscale
        simple_imgnext = imgsimplify(img_next, c_in, window_next)

        # Compute NCC considering the previous frames as templates
        NCC_max = 0
        for template in previous_imgs[0:min(i, len(previous_imgs))]:
            NCC_obtained = match_template(
                simple_imgnext, template, pad_input=True, mode="edge")
            NCC_obtained_max = np.max(NCC_obtained)
            # Take highest NCC value of all templates
            if NCC_obtained_max > NCC_max:
                NCC_vals = NCC_obtained
                NCC_max = NCC_obtained_max

        # Take the coordinates of the pixel with the highest NCC value
        candidate_pixel = np.argmax(NCC_vals)
        candidate_pixel = np.unravel_index(candidate_pixel, NCC_vals.shape)
        h_fin, w_fin = candidate_pixel[0], candidate_pixel[1]

        # Update initial conditions
        c_in = (c_in[0] - (window_next - 1) // 2 + w_fin,
                c_in[1] - (window_next - 1) // 2 + h_fin)
        previous_imgs[i % num_imgs] = imgsimplify(img_next, c_in,
                                                  window_template)

        # Save new coordinates in the .txt file
        output.write("{}\t\t{}\t\t{}\n".format(files_names[i], c_in[0],
                                               c_in[1]))
        output.flush()

        #Visual check of the obtained results
        plt.ion()
        plt.clf()
        plt.imshow(img_to_plot)
        plt.scatter(*c_in, c="r", marker="o")
        plt.show()
        plt.pause(0.2)

    output.close()
