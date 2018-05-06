#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 06:30:32 2018

@author: Shushan

This script reads the images provided in the directory into a 4D numpy array.
To use it simply call the function StoreImages and give 1 argument - the absolute path
of the directory
"""


import numpy as np
import os
from scipy.misc import imread
from os.path import isfile, join



def getImageNames(imgPath):
    return [f for f in os.listdir(imgPath) if isfile(join(imgPath, f))]


DATA_DIR = 'C:\\Users\\Shushan\\Desktop\\Mission Masters\\ISIP\\Group Project\\project_data\\project_data\\a'

def StoreImages(directory):
    img_names = getImageNames(directory) #extract image file names into a list
    img0 = imread(os.path.join(directory, img_names[0]), mode='RGB') #read the first image
    dim1, dim2, dim3 = img0.shape #get the dimensions of the result array
    img_frames = np.zeros(shape=(dim1, dim2, dim3, len(img_names))) #create an empty array
    for index, img in enumerate(img_names): #loop over images and get the count of the image too
        img_frames[:,:,:,index] = imread(os.path.join(directory, img), mode='RGB') # save the 3D array in the correspoinding 4th D index
    return img_frames #return the 4D array



#if __name__ == "__main__":
#    main()
