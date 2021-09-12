import cv2
import numpy as np
import os
import random, string
import math

from pdb import set_trace as bp

def create_noisy_gt(image,output_downscale=8,blur_sigma=1):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, 225, 250)
    assert(output_downscale == 8)
    gray = cv2.resize(gray,(gray.shape[1]//output_downscale, gray.shape[0]//output_downscale))

    kernal_size_from_actual = 5
    blur = cv2.GaussianBlur(gray,(kernal_size_from_actual,kernal_size_from_actual),sigmaX = blur_sigma)

    orig_blur = blur.copy()
    blur = blur.astype('float32') / 255
    blur = blur / 10

    return blur
