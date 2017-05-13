# -*- coding: utf-8 -*-
"""
Signal and Image Processing 2017 by Raphael Sznitman.

Team Project
- Livio Baetscher
- Manuela Haefliger
- Marc-Antoine Jacques

Python Version: 3.6.1
"""

"""
- Need to build a library of patterns: one full, one torso, one head...
- Return the position with the highest score in all arrays

Parameters:
- template
- pyramid nber of layers
- pyramid downscale factor (reduction of dimension at each successive pyramid level)

"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import scipy.signal
import time
from skimage import filters


def findwaldo(img, template):
    """
    Returns the position of Waldo in image
    :param img: path to image file
    :param template: path to image file representing the researched pattern
    :return: a 2-tuple corresponding to the position of Waldo
    """
    # 1) Read images
    img = plt.imread(img)
    template = plt.imread(template)
    # 2) Gaussian pyramid img (tuple of arrays)
    pyramid = tuple(skimage.transform.pyramid_gaussian(img, downscale=2))
    del img  # only use pyramid
    # 3) Convolution with template
    out = scipy.signal.convolve2d(pyramid[0][..., 0], template[..., 0], mode='same')
    # 4) Noise removal?

    return out


i = plt.imread('./data/images/27.jpg')
t = plt.imread('./template/stripes2.jpg')
pi = tuple(skimage.transform.pyramid_gaussian(i, downscale=2))
pt = tuple(skimage.transform.pyramid_gaussian(t, downscale=2))

t0 = time.time()
temp = scipy.signal.convolve2d(pi[0][..., 0], pt[4][..., 0], mode='same')
temp2 = scipy.signal.convolve2d(pi[0][..., 1], pt[4][..., 1], mode='same')
temp3 = scipy.signal.convolve2d(pi[0][..., 2], pt[4][..., 2], mode='same')
#temp = findwaldo('./data/images/01.jpg', './template/full.png')
t1 = time.time()
print('Ellapsed time: {:02f}'.format(t1-t0))

temp4 = temp + temp2 + temp3
plt.figure()
plt.imshow(temp4)
plt.colorbar()
plt.show()
np.where(temp4 == temp4.max())


# =================================================
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


# =================================================
# Keep only red
i2 = np.copy(i)
i2 = rgb2gray(i2)


# Create a red and white stripe pattern
stripes = np.zeros((70, 90))
# Create white stripes
stripes[0:10, :] = 255
stripes[20:30, :] = 255
stripes[40:50, :] = 255
stripes[60:70, :] = 255
plt.imshow(stripes, cmap='gray'); plt.colorbar()


t2 = np.copy(t)
t2[np.where(t2[..., 0] <= 200)] = 0
t2[np.where(t2[..., 1] >= 100)] = 0
t2[np.where(t2[..., 2] >= 100)] = 0
t2 = rgb2gray(t2)

pi = tuple(skimage.transform.pyramid_gaussian(i2, downscale=2))
pt = tuple(skimage.transform.pyramid_gaussian(t2, downscale=2))

t0 = time.time()
temp = scipy.signal.convolve2d(pi[0], stripes, mode='same')
#temp = findwaldo('./data/images/01.jpg', './template/full.png')
t1 = time.time()
print('Ellapsed time: {:02f}'.format(t1-t0))
plt.imshow(temp); plt.colorbar()

plt.figure()
plt.imshow(pi[0])
plt.imshow(temp, alpha=0.5)
plt.show()
