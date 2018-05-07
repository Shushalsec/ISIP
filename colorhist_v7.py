#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.feature import match_template
from skimage.color import rgb2gray


def imgsimplify(img, c, window):
    """
    Returns the part of the image that might be of interest
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
    c_x = c[0]
    c_y = c[1]
    window_half = (window - 1) // 2
    simple_img = img[c_y-window_half:c_y+window_half+1, c_x-window_half:c_x+window_half+1]
    return simple_img

if __name__== '__main__':
    #_______________________________________________________________
    # Activate to work only with set A
    # files_names = glob("./project_data/a/*.png")
    # c_in = 348, 191
    # output = open('output_a.txt', 'w')
    # training_pictures = glob("./project_data/a_train/ncc/*.png")
    # c_train = []
    #_______________________________________________________________
    # Activate to work only with set B
    files_names = glob("./project_data/b/*.png")
    c_in = 439, 272
    output = open('output_b.txt', 'w')
    training_pictures = glob("./project_data/b_train/ncc/*.png")
    c_train = [(433, 270), (426, 263)]
    #_______________________________________________________________

    files_names.sort()
    training_pictures.sort()

    # Parameters
    window_in = 35
    window_next = window_in * 2 + 1
    threshold = 0.98

    # Initial conditions
    img = rgb2gray(plt.imread(files_names[0]))
    simple_img = imgsimplify(img, c_in, window_in)

    # Training information
    train_info = np.zeros((len(training_pictures), window_in, window_in))
    for j in range(len(training_pictures)):
        img_train = rgb2gray(plt.imread(training_pictures[j]))
        train_info[j,:,:] = imgsimplify(img_train, c_train[j], window_in)

    # Open a .txt file where the coordinates of the points will be writen
    output.write("  image_name\t    x-location\t    y-location\n")
    output.write("{}\t\t{}\t\t{}\n".format(files_names[0], c_in[0], c_in[1]))
    output.flush()

    # Find the desired pixel in all frames
    for i in range(1,len(files_names)):
        # Take next image of the list for comparison and its "interesting" regions
        img_next = rgb2gray(plt.imread(files_names[i]))
        simple_imgnext = imgsimplify(img_next, c_in, window_next)

        # Compute NCC and take the pixel with the highest value
        NCC_onedim = match_template(simple_imgnext, simple_img, pad_input=True, mode="edge")
        NCC_max = np.max(NCC_onedim)

        # Compute NCC (same as before) with a more relaxed template (a bigger one)
        template_amplified = imgsimplify(img, c_in, window_in + 10)
        NCC_onedim_ampl = match_template(simple_imgnext, template_amplified, pad_input=True, mode="edge")
        NCC_max_ampl = np.max(NCC_onedim_ampl)

        # Check which template is better (the one with higher NCC)
        if NCC_max_ampl > NCC_max:
            NCC_onedim = NCC_onedim_ampl
            NCC_max = NCC_max_ampl

        # Check if NCC_max is still lower than the threshold.
        # If so, use the training images
        if NCC_max < threshold:
            NCC_max = 0
            for training in train_info:
                NCC_training_onedim = match_template(simple_imgnext, training, pad_input=True, mode="edge")
                NCC_train_max = np.max(NCC_training_onedim)
                if  NCC_train_max > NCC_max:
                    NCC_max = NCC_train_max
                    NCC_onedim = NCC_training_onedim

        poss_pix_NCC = np.argmax(NCC_onedim)
        poss_pix_NCC = np.unravel_index(poss_pix_NCC, NCC_onedim.shape)
        h_fin, w_fin = poss_pix_NCC[0], poss_pix_NCC[1]

        # Update initial conditions
        img = img_next
        c_in = (c_in[0] - (window_next - 1)//2 + w_fin, c_in[1] - (window_next - 1)//2 + h_fin)
        simple_img = imgsimplify(img_next, c_in, window_in)

        # Save new coordinates in the .txt file
        output.write("{}\t\t{}\t\t{}\n".format(files_names[i], c_in[0], c_in[1]))
        output.flush()

        #Visual check of the obtained results
        plt.ion()
        plt.clf()
        plt.imshow(img, cmap = plt.get_cmap('gray'))
        plt.scatter(*c_in, c="r", marker="o")
        plt.show()
        plt.pause(0.2)

    output.close()
