# -*- coding: utf-8 -*-
"""
Signal and Image Processing 2017 by Raphael Sznitman.

Team Project
- Livio Baetscher
- Manuela Haefliger
- Marc-Antoine Jacques

Python Version: 3.6.1
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import scipy.signal
import time
from skimage import filters, feature
from math import pi


def rgb2gray(rgb):
    """
    Convert an RGB image to grayscale according to https://en.wikipedia.org/wiki/Grayscale
    :param rgb: a 3D numpy array
    :return: a 2D numpy array
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def PlotHeatmap(image, score_image, alpha = 0.7, title = "", bar = True):
    """
    Overlay two images, score_image with alpha transparency
    :param image: numpy array
    :param score_image: numpy array  
    :param alpha: transparency
    :return: "Heatmap plot"
    """
    plt.figure()
    plt.imshow(image)
    plt.imshow(score_image, alpha=alpha)
    plt.title(title)
    if bar:
        plt.colorbar()
    plt.show()


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
    image[image <= threshold] = 255
    image[image >= threshold] = 0
    return image


def StripeMotif(height, width, nber_stripe=4):
    """
    Create a grayscale stripes pattern, representing Waldo's shirt 
    :param width: width of the motif
    :param height: height of the motif
    :param nber_stripe: number of partitions
    :return: 2D numpy array, binarized with 0 and 255
    """
    if height % nber_stripe != 0:
        raise ValueError('height must be a multiple of nber_stripe')
    stripes = np.zeros((height, width))
    stripe_height = int(height/nber_stripe)
    breakpoints = [(i, i+stripe_height) for i in range(0, height - stripe_height + 1, stripe_height*2)]
    for rows in breakpoints:
        stripes[rows[0]:rows[1], :] = 255
    return stripes


def ExtractRed(image, minimum_red = 150, threshold_green = 100, threshold_blue = 100):
    """
    Isolate red color from an image
    :param img: a 3D numpy array
    :return: a 3D numpy array with red zones
    """

    # Suppress all parts where green and blue are too strong and red too low (remoive non-red)
    img = np.copy(image)
    img[np.where(img[..., 1] >= threshold_green)] = 0
    img[np.where(img[..., 2] >= threshold_blue)] = 0
    img[np.where(img[..., 0] <= minimum_red)] = 0

    # Set all red pixels to 255
    img[np.where(img[..., 0] > 0)] = 255
    return img


def DrawRectangle(image, x, y, box_size):
    """
    Set values around coordinates x,y to 255. Useful for displaying area around peaks of signal.
    :param image: 2D numpy array
    :param x: int
    :param y: int
    :param box_size: size of the box
    :return: 2D numpy array
    """
    image[(x-box_size):(x+box_size), (y-box_size):(y+box_size)] = True


# ===============================================================
# =                       One example                           =
# ===============================================================
"""
Idea:
convolve2d
correlate2d
fftconvolve (much faster, similar results than convolve2d)

gabor filter: http://matlabserver.cs.rug.nl/edgedetectionweb/web/edgedetection_params.html, http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/SIKLOSSY/bars.html
"""

# Parameters
image = './data/images/04.jpg'
min_red, max_green, max_blue = 200, 100, 100
min_dist_peak, thresh_peak, max_nber_peak = 20, 0.2, 5
size_box = 10


# Read Image and returns a binarized greyscale image with red pixels only
image = plt.imread(image)
reds = ExtractRed(image, min_red, max_green, max_blue)
reds_grayscale = rgb2gray(reds)
PlotHeatmap(image, reds, title='Binarized Red Pixels map', bar=False)

# Shirt of Waldo, /!\ if use a non-symmetric template, flip it if use convolution (so that it does the same as correlation)
stripe_template = reds_grayscale[1170:1191, 1143:1151]
stripe_template = np.fliplr(stripe_template)
stripe_template = np.flipud(stripe_template)
#stripe_template = StripeMotif(height=16, width=3, nber_stripe=4)

t1 = time.time()
# Look for template, heatmap of template in different regions
score = scipy.signal.fftconvolve(reds_grayscale, stripe_template, mode='same')
# Isolate peaks
peak_positions = feature.corner_peaks(score, min_distance=min_dist_peak, indices=True, threshold_rel=thresh_peak, num_peaks=max_nber_peak)
t2 = time.time()
print('Elapsed time: {:03f}'.format(t2-t1))
PlotHeatmap(image, score, title='Convolution score')

# Draw a rectangle at the position of the peaks
peak_positions_img = feature.corner_peaks(score, min_distance=min_dist_peak, indices=False, threshold_rel=thresh_peak, num_peaks=max_nber_peak)
for pos in peak_positions:
    DrawRectangle(peak_positions_img, pos[0], pos[1], size_box)
PlotHeatmap(image, peak_positions_img, title='Most probable positions of Waldo', bar=False)


# =================================
# DAnker: warzone :/
# ToDo: Set the template for the glasses (Yes we hope this will help).
# Note: the preprocessing is no longer used.
# Parameters
image = './data/images/04.jpg'
min_dist_peak, thresh_peak, max_nber_peak = 20, 0.2, 5
size_box = 10

# Read Image and returns a binarized greyscale image with red pixels only
image = plt.imread(image)

# Glasses of Waldo, /!\ if use a non-symmetric template, flip it if use convolution (so that it does the same as correlation)
cp = np.copy(image)
stripe_template = cp[1146:1154, 1144:1155]
stripe_template = np.fliplr(stripe_template)
stripe_template = np.flipud(stripe_template)

t1 = time.time()
# Look for template, heatmap of template in different regions
score = scipy.signal.fftconvolve(image, stripe_template, mode='same')
# Isolate peaks
peak_positions = feature.corner_peaks(score, min_distance=min_dist_peak, indices=True, threshold_rel=thresh_peak, num_peaks=max_nber_peak)
t2 = time.time()
print('Elapsed time: {:03f}'.format(t2-t1))
PlotHeatmap(image, score, title='Convolution score')
peak_positions_img = feature.corner_peaks(score, min_distance=min_dist_peak, indices=False, threshold_rel=thresh_peak, num_peaks=max_nber_peak)
for pos in peak_positions:
    DrawRectangle(peak_positions_img, pos[0], pos[1], size_box)
PlotHeatmap(image, peak_positions_img, title='Most probable positions of Waldo', bar=False)
