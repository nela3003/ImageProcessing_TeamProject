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
import scipy.signal
import skimage.feature


def find_waldo(image):
    """
    Look for Waldo's shirt and beanie in red and white pixels.
    Return a single pos (x,y) for most probable Waldo's position, where (0,0) states the lower left corner of the image.
    :param image: an numpy array of shape (n, m, 3)
    :return: a tuple of 2 integers representing Waldo's position
    """

    # Define how many peaks from each filter should be considered
    nber_peaks = 1

    # Define minimum distance between 2 peaks intra and inter filters.
    # Intra: if 2 peaks within the range are detected, no peak is returned.
    # Inter: when looking for identical peaks across filters, 2 peaks within the range are combined into a single one.
    min_dist_peaks = 100

    # Define minimum value of red and maximum value of blue and green respectively,
    # to classify a pixel as red in preprocessing color extraction.
    extract_red = (150, 100, 100)

    # Define minimum value of red, blue and green respectively,
    # to classify a pixel as white in preprocessing color extraction
    extract_white = (220, 220, 180)

    # Create the filter bank, a list of (x, y, 2) np.arrays as returned by StripeMotifRW
    filter_bank = HatShirtBankRW(shirt_stripe_nber=4)

    # Put call parameters in convenient variables
    rr, rg, rb = extract_red
    wr, wg, wb = extract_white

    reds = ExtractRed(image, rr, rg, rb)
    grayscale_reds = rgb2gray(reds)
    grayscale_reds -= np.mean(grayscale_reds)

    whites = ExtractWhite(image, wr, wb, wg)
    grayscale_whites = rgb2gray(whites)
    grayscale_whites -= np.mean(grayscale_whites)

    # Create a (x, y, 2) RED-WHITE image
    rw_image = np.empty((image.shape[0], image.shape[1], 2))
    rw_image[..., 0] = grayscale_reds
    rw_image[..., 1] = grayscale_whites

    # Apply filter banks, each response is composed of a red and white channel, so response is number_filter*(x, y, 2) list
    response = []
    for filt in filter_bank:
        # Flip the filter to make convolution behaves as correlation, remove mean to center the filter
        template = filt.copy()
        template = np.fliplr(template)
        template = np.flipud(template)
        template -= int(np.mean(template))
        # Perform convolution and normalize the score output
        score = scipy.signal.fftconvolve(rw_image, template, mode='same')
        score = (score - score.mean()) / score.std()
        response.append(score)

    # Get the peaks coordinates of each filter response
    all_peaks = []
    for i in range(len(response)):
        reds_peak_positions = skimage.feature.corner_peaks(response[i][..., 0], min_distance=min_dist_peaks,
                                                           indices=True, threshold_rel=0.2, num_peaks=nber_peaks)
        whites_peak_positions = skimage.feature.corner_peaks(response[i][..., 1], min_distance=min_dist_peaks,
                                                             indices=True, threshold_rel=0.2, num_peaks=nber_peaks)
        all_peaks.append(reds_peak_positions)
        all_peaks.append(whites_peak_positions)

    all_peaks_intensities = []
    for i in range(len(response)):
        all_peaks_intensities.append([response[i][peak[0], peak[1], 0] for peak in all_peaks[2*i]])
        all_peaks_intensities.append([response[i][peak[0], peak[1], 1] for peak in all_peaks[2*i + 1]])

    # Peak selection
    conc_all_peaks = np.concatenate(all_peaks)
    conc_all_peaks_intensities = np.concatenate(all_peaks_intensities)
    dist = scipy.spatial.distance.pdist(conc_all_peaks)
    dist_sq = np.triu(scipy.spatial.distance.squareform(dist))
    tmps = np.where(np.logical_and(dist_sq < min_dist_peaks, dist_sq != 0))

    if len(tmps[0]) != 0:
        idx = np.unique(np.concatenate(tmps))
        # In case of multiple peaks counted at least twice, go to intensity and pick the peak with highest intensity
        conc_all_peaks = conc_all_peaks[idx]
        conc_all_peaks_intensities = conc_all_peaks_intensities[idx]

    y, x = conc_all_peaks[np.argmax(conc_all_peaks_intensities)]

    return x, image.shape[0]-y


def HatShirtBankRW(shirt_stripe_thickness=[3, 4, 5, 6], shirt_stripe_nber=6, distance_hat_shirt=1.5, angle=45):
    bank = []
    for thick in shirt_stripe_thickness:
        bank.append(HatShirtMotifRW(thick, shirt_stripe_nber, distance_hat_shirt, angle))
    return bank


def HatShirtMotifRW(shirt_stripe_thickness, shirt_stripe_nber, distance_hat_shirt=1.5, angle=45):
    from scipy.ndimage.interpolation import rotate
    # Create a square hat with 2 red stripes
    hat_stripe_thickness = int(0.7373 * shirt_stripe_thickness - 0.4249)
    hat = StripeMotifRW(3*hat_stripe_thickness, 3*hat_stripe_thickness, 3)
    hat = rotate(hat, angle)
    # Create shirt
    shirt = StripeMotifRW(shirt_stripe_thickness*shirt_stripe_nber, hat.shape[1], shirt_stripe_nber)
    # Fill the template, hat at the top, shirt at the bottom
    out = np.zeros((hat.shape[0] + shirt.shape[0] + int(distance_hat_shirt*hat.shape[0]), hat.shape[1], 2))
    out[:hat.shape[0], :hat.shape[1], 0] = hat[..., 0]
    out[:hat.shape[0], :hat.shape[1], 1] = hat[..., 1]
    out[int(hat.shape[0]*(1 + distance_hat_shirt)):, :, 0] = shirt[..., 0]
    out[int(hat.shape[0] * (1 + distance_hat_shirt)):, :, 1] = shirt[..., 1]
    # Binarize to compensate interpolation of rotation
    out[out >= 100] = 255
    out[out < 100] = 0
    return out


def StripeMotifRW(height, width, nber_stripe=4):
    if height % nber_stripe != 0:
        raise ValueError('height must be a multiple of nber_stripe')
    stripes_red = np.zeros((height, width))
    stripes_white = np.zeros((height, width))
    stripe_height = int(height/nber_stripe)

    breakpoints = [(i, i+stripe_height) for i in range(0, height - stripe_height + 1, stripe_height*2)]
    for rows in breakpoints:
        stripes_red[rows[0]:rows[1], :] = 255

    pos = np.where(stripes_red != 255)
    stripes_white[pos] = 255
    stripes_rw = np.empty((height, width, 2))
    stripes_rw[..., 0] = stripes_red
    stripes_rw[..., 1] = stripes_white
    return stripes_rw


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
    """
    Isolate white color from an image
    :param img: a 3D numpy array
    :return: a 3D numpy array with white zones
    """
    img = np.copy(image)
    img[np.where(img[..., 1] <= minimum_green)] = 0
    img[np.where(img[..., 2] <= minimum_blue)] = 0
    img[np.where(img[..., 0] <= minimum_red)] = 0

    # Set all white pixels to 255
    img[np.where(img[..., 0] > 0)] = 255
    return img


def rgb2gray(rgb):
    """
    Convert an RGB image to grayscale according to https://en.wikipedia.org/wiki/Grayscale
    :param rgb: a 3D numpy array
    :return: a 2D numpy array
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
