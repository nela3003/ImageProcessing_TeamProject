# -*- coding: utf-8 -*-
"""
Signal and Image Processing 2017 by Raphael Sznitman.

Team Project
- Livio Baetscher
- Manuela Haefliger
- Marc-Antoine Jacques

Python Version: 3.6.1

Isolate red channel and remove as much background as possible, make the shirt stripes motif as clean as possible
"""


import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import scipy.signal
import time
from skimage import filters


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def RemoveBackground(img, method, window_size = 25, k = 0.8):
    """
    Create a binary image separating foreground from background
    :param img: a numpy array
    :param method: one of ['otsu', 'niblack', 'sauvola']
    :param window_size: size of neighborhood used to define threshold, used in ['niblack', 'sauvola']
    :param k: used to tune local threshold, used in ['niblack', 'sauvola']
    :return: numpy array, representing a binary image
    """
    image = np.copy(img)
    if method == 'otsu':
        threshold = filters.threshold_otsu(img)
    elif method == 'niblack':
        threshold = filters.threshold_niblack(img, window_size, k)
    elif method == 'sauvola':
        threshold = filters.threshold_sauvola(img)
    #image[image <= threshold] = 255
    image[image >= threshold] = 0
    return image


def StripeMotif(height, width=16*height/9,  nber_stripe):
    """
    # Create a grayscale stripes pattern, representing Waldo's shirt 
    :param width: width of the motif
    :param height: height of the motif
    :param nber_stripe: number of partitions
    :return: 2D numpy array, binarized with 0 and 255
    """
    stripes = np.zeros((height, width))
    stripe_height = height/nber_stripe
    breakpoints = [(i, i+stripe_height) for i in range()]
    stripes[0:10, :] = 255
    stripes[20:30, :] = 255
    stripes[40:50, :] = 255
    stripes[60:70, :] = 255



i = plt.imread('./data/images/27.jpg')
t = plt.imread('./template/stripes2.jpg')
pi = tuple(skimage.transform.pyramid_gaussian(i, downscale=2))
pt = tuple(skimage.transform.pyramid_gaussian(t, downscale=2))

plt.imshow(i[...,0]); plt.colorbar()  # Red channel
i2 = np.copy(i)
i2[np.where(i2[..., 1] >= 100)] = 0
i2[np.where(i2[..., 2] >= 100)] = 0



i2 = rgb2gray(i2)
plt.imshow(i2)

temp = RemoveBackground(i2, 'sauvola', window_size=25, k=0.8)
plt.figure()
plt.subplot(121)
plt.imshow(i2, cmap = 'gray')
plt.subplot(122)
plt.imshow(temp, cmap = 'gray')
plt.show()
del temp