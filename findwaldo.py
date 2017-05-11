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


i = plt.imread('./data/images/01.jpg')
t = plt.imread('./template/head.png')
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
