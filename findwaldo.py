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
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform

def findwaldo(img, template):
    """
    Returns the position of Waldo in image
    :param img: path to image file
    :param template: a numpy array (image) representing the researched pattern
    :return: a 2-tuple corresponding to the position of Waldo
    """
    # 1) Read image

    # 2) Gaussian pyramid img

    # 3) Convolution with template

    # 4) Noise removal?

    return x, y
