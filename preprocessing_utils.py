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
import scipy.spatial.distance


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


def CombinePeaks2(width, height, conc_all_peaks, conc_all_peaks_intensities, min_dist_peaks=100):
    peak_map = np.zeros((height, width))
    for i in range(len(conc_all_peaks)):
        peak_map[conc_all_peaks[i][0],conc_all_peaks[i][1]] = conc_all_peaks_intensities[i]
    conc_unique_peaks = skimage.feature.peak_local_max(peak_map, min_distance=50)
    # For each unique peak, return mean signal of this peak
    conc_unique_intensities = []
    for peak in conc_unique_peaks:
        # Area around the peak
        area = peak_map[peak[0]-int(min_dist_peaks/2):peak[0]+int(min_dist_peaks/2),
               peak[1]-int(min_dist_peaks/2):peak[1]+int(min_dist_peaks/2)]
        conc_unique_intensities.append(np.mean(area[np.nonzero(area)]))
    # conc_unique_intensities = [peak_map[pos[0], pos[1]] for pos in conc_unique_peaks]
    return conc_unique_peaks, conc_unique_intensities



def CombinePeaks(peakspos, intensities, min_dist_peaks=150):
    """
    Combine close peaks (within euclidean distance < min_dist_peaks) as coordinate average and maximum intensity.
    :param peakspos: list of lists of peak coordinates [[(x1,y1),(x2,y2)],[(x3,y3),(x4,y4),(x5,y5)]]
    :param intensities: list of lists of peak intesities [[z1,z2],[z3,z4,z5]]
    :param min_dist_peaks:
    :return unique_peakspos_new: new coordinates of the peaks
    :return unique_intensities_new: new intentsities for the peaks
    """
    unique_peakpos_new = np.concatenate(peakspos)
    unique_intensities_old = np.concatenate(intensities)
    unique_peakpos_old = []
    unique_intensities_new = []
    while not np.array_equal(unique_peakpos_old, unique_peakpos_new):
        unique_peakpos_old = unique_peakpos_new.copy()
        dist = scipy.spatial.distance.pdist(unique_peakpos_old)
        dist_sq = np.triu(scipy.spatial.distance.squareform(dist))
        tmps = np.where(np.logical_and(dist_sq < min_dist_peaks, dist_sq != 0))
        candidates = [(tmps[0][i], tmps[1][i]) for i in range(len(tmps[0]))]
        unique_peakpos_new = [np.mean([unique_peakpos_old[x[0]], unique_peakpos_old[x[1]]], axis=0) for x in candidates]
        unique_intensities_new = [np.max([unique_intensities_old[x], unique_intensities_old[y]]) for x, y in candidates]
        for i in range(len(unique_peakpos_old)):
            if i not in np.unique(np.concatenate(candidates)):
                unique_peakpos_new.append(unique_peakpos_old[i])
                unique_intensities_new.append(unique_intensities_old[i])
        unique_peakpos_new = [x.astype(int) for x in unique_peakpos_new]
    return unique_peakpos_old, unique_intensities_new

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

