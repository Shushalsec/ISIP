#!/usr/bin/env python3

# INTRODUCTION TO SIGNAL AND IMAGE PROCESSING 2018 - GROUP PROJECT

# TRACKING SURGICAL INSTRUMENTS USING NCC

# Shushan Toneyan, Madleina Caduff, Judith Bergadà Pijuan

'''
In lines 42-43, it can be specified whether a text file should be written and/or a
video of the output should be produced (which will be saved in your working directory).
'''

from glob import glob
import cv2  # we use openCV only for making the final video!
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_template


def imgsimplify(img, c, window):
    """
    Returns the part of the image
    :param img: array_like; image that we want to cut or simplify
    :param c: tuple; center of the new image
    :param window: odd integer; size of the desired final image.
    :return: array_like; image of size window x window that results from cutting the initial img.
    """
    c_x, c_y = c[0], c[1]
    window_half = (window - 1) // 2
    simple_img = img[c_y - window_half: c_y + window_half + 1,
                     c_x - window_half: c_x + window_half + 1]
    return simple_img


if __name__ == '__main__':

    # Define output:
    whichSet = 'A'
    makeVideo = True
    writeTextFile = True

    if whichSet == 'A':
        files_names = glob("./project_data/a/*.png")
        c_in = (348, 191)
        if writeTextFile:
            output = open('output_a.txt', 'w')
        video_name = "video_a.mp4"
    if whichSet == 'B':
        files_names = glob("./project_data/b/*.png")
        c_in = (439, 272)
        if writeTextFile:
            output = open('output_b.txt', 'w')
        video_name = "video_b.mp4"
    else:
        print('Set name not defined, try again.')

    files_names.sort()

    # Parameters
    num_imgs = 15
    window_next = 35
    window_template = window_next // 2

    # Initial conditions
    img = rgb2gray(plt.imread(files_names[0]))
    previous_imgs = np.zeros((num_imgs, window_template, window_template))
    previous_imgs[0, :, :] = imgsimplify(img, c_in, window_template)

    # Open a .txt file where the coordinates of the points will be writen
    if writeTextFile:
        output.write("  image_name\t    x-location\t    y-location\n")
        output.write("{}\t\t{}\t\t{}\n".format(files_names[0], c_in[0], c_in[1]))
        output.flush()

    # Open the video to save the results
    if makeVideo:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(video_name, fourcc, 5, (img.shape[1], img.shape[0]))

    # Find the desired pixel in all frames
    for i in range(1, len(files_names)):
        # Take next image of the list and its potential regions to find the point
        img_to_plot = plt.imread(files_names[i])
        img_next = rgb2gray(img_to_plot)  # convert to grayscale
        simple_imgnext = imgsimplify(img_next, c_in, window_next)

        # Compute NCC considering the previous frames as templates
        NCC_max = 0
        for template in previous_imgs[0: min(i, len(previous_imgs))]:
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

        # Save new frame in the video
        if makeVideo:
            img_to_plot = (img_to_plot * 255).astype(np.uint8)
            cv2.circle(img_to_plot, c_in, 0, (255, 0, 0), 9)
            video_writer.write(img_to_plot[:, :, ::-1])

        # Save new coordinates in the .txt file
        if writeTextFile:
            output.write("{}\t\t{}\t\t{}\n".format(files_names[i], c_in[0],
                                                   c_in[1]))
            output.flush()

    if writeTextFile:
        output.close()
    if makeVideo:
        video_writer.release()
