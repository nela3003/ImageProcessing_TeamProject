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
Idea:
convolve2d
correlate2d
fftconvolve (much faster, similar results than convolve2d)

gabor filter: http://matlabserver.cs.rug.nl/edgedetectionweb/web/edgedetection_params.html, http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/SIKLOSSY/bars.html
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import scipy.signal
import time
from skimage import filters, feature, morphology
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


def HatMotif(angle, pompom_radius, white_thick, red_thick, black_thick, width=2):
    from scipy.ndimage.interpolation import rotate
    height = pompom_radius + white_thick + red_thick + 2*black_thick
    hat = np.zeros((height, width))
    hat[:(pompom_radius + 1), :] = 255
    hat[pompom_radius+2*black_thick+white_thick:, :] = 255
    hat = rotate(hat, angle, cval=0)
    hat[hat >= 100] = 255
    hat[hat < 100] = 0
    return hat


def HatShirtMotif(shirt_stripe_thickness, shirt_stripe_nber, distance_hat_shirt=1.5, angle=45):
    """
    Create a whole Waldo model in red stripes
    :param shirt_stripe_thickness: integer, red stripe thickness of the shirt
    :param shirt_stripe_nber: integer, number stripes in shirt
    :param distance_hat_shirt: float, distance between bottom of hat and top of shirt, distance in hat height unit
    :param angle: float, angle of the hat compared to horizontal
    :return: a 2D binarized np array
    """
    from scipy.ndimage.interpolation import rotate
    # Create a square hat with 2 red stripes
    hat_stripe_thickness = int(0.7373 * shirt_stripe_thickness - 0.4249)
    hat = StripeMotif(3*hat_stripe_thickness, 3*hat_stripe_thickness, 3)
    hat = rotate(hat, angle)
    # Create shirt
    shirt = StripeMotif(shirt_stripe_thickness*shirt_stripe_nber, hat.shape[1], shirt_stripe_nber)
    # Fill the template, hat at the top, shirt at the bottom
    out = np.zeros((hat.shape[0] + shirt.shape[0] + int(distance_hat_shirt*hat.shape[0]), hat.shape[1]))
    out[:hat.shape[0], :hat.shape[1]] = hat
    out[int(hat.shape[0]*(1 + distance_hat_shirt)):, :] = shirt
    # Binarize to compensate interpolation of rotation
    out[out >= 100] = 255
    out[out < 100] = 0
    return out



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


def ExtractWhite(image, minimum_red=230, minimum_green=230, minimum_blue=230):
    img = np.copy(image)
    img[np.where(img[..., 1] <= minimum_green)] = 0
    img[np.where(img[..., 2] <= minimum_blue)] = 0
    img[np.where(img[..., 0] <= minimum_red)] = 0

    # Set all white pixels to 255
    img[np.where(img[..., 0] > 0)] = 255
    return img


def ExtractBlack(image, threshold_red = 80, threshold_green = 80, threshold_blue = 80):
    """
    Isolate black color from an image
    :param img: a 3D numpy array
    :return: a 3D numpy array with red zones
    """

    # Suppress all parts where green and blue are too strong and red too low (remoive non-red)
    img = np.copy(image)
    img[np.where(img[..., 1] >= threshold_green)] = 0
    img[np.where(img[..., 2] >= threshold_blue)] = 0
    img[np.where(img[..., 0] >= threshold_red)] = 0

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


def FindMaximaListArray(response_array):
    """
    Go through a list of arrays and returns the maximum value and its position
    :param response_array: a list of 2D-arrays
    :return: 
    """
    max_sf = 0
    loc_sf = None  # [x, y, array index]
    arr_sf = None
    i = 0
    for arr in response_array:
        if arr.max() > max_sf:
            max_sf = arr.max()
            loc_sf = np.unravel_index(arr.argmax(), arr.shape)
            arr_sf = i
        i += 1
    return loc_sf, arr_sf


################## K-Means ##################
# To Replace Extract Red and Extract White?
"""http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin
w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))
n_colors = 64
image_array_sample = shuffle(image_array, random_state=0)[:5000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)
codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(image)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()
"""