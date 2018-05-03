# -*- coding: utf-8 -*-
"""
Created on Thu May  3 06:30:32 2018

@author: Shushan
"""

import numpy as np
import os
from scipy.misc import imread
from os.path import isfile, join
import matplotlib.pyplot as plt



def getImageNames(imgPath):
    return [f for f in os.listdir(imgPath) if isfile(join(imgPath, f))]


file_dir_a = 'project_data\\project_data\\a'
cwd = os.getcwd()

DATA_DIR = os.path.join(cwd, file_dir_a)





def main():
    img_names = getImageNames(DATA_DIR)
    img_frames = np.zeros(shape=(480, 640, 3, len(img_names)))
    for index, img in enumerate(img_names):
        img_frames[:,:,:,index] = imread(os.path.join(DATA_DIR, img), mode='RGB')

if __name__ == "__main__":
    main()
