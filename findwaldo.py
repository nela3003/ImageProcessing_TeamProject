# -*- coding: utf-8 -*-
"""
Signal and Image Processing 2017 by Raphael Sznitman.

Team Project
- Livio Baetscher
- Manuela Haefliger
- Marc-Antoine Jacques

Python Version: 3.6.1
"""

from preprocessing_utils import *

def find_waldo(image):
    """
    Look for Waldo's shirt and beanie in red and white pixels.
    Return a single pos (x,y) for most probable Waldo's position, where (0,0) states the lower left corner of the image.
    :param image: an numpy array of shape (n, m, 3)
    :return: a tuple of 2 integers representing Waldo's position
    """

    # Define how many peaks from each filter should be considered
    nber_peaks = 5

    # Define minimum distance between 2 peaks intra and inter filters.
    # Intra: if 2 peaks within the range are detected, no peak is returned.
    # Inter: when looking for identical peaks across filters, 2 peaks within the range are combined into a single one.
    min_dist_peaks = 150

    # Define minimum value of red and maximum value of blue and green respectively,
    # to classify a pixel as red in preprocessing color extraction.
    extract_red = (150, 100, 100)

    # Define minimum value of red, blue and green respectively,
    # to classify a pixel as white in preprocessing color extraction
    extract_white = (220, 220, 180)

    # Create the filter bank, a list of (x, y, 2) np.arrays as returned by StripeMotifRW
    # TODO: filter_bank

    # Put call parameters in convenient variables
    rr, rg, rb = extract_red
    wr, wg, wb = extract_white

    # Preprocessing steps
    image = plt.imread(image)

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
        reds_peak_positions = feature.corner_peaks(response[i][..., 0], min_distance=min_dist_peaks, indices=True,
                                                   threshold_rel=0.2, num_peaks=nber_peaks)
        whites_peak_positions = feature.corner_peaks(response[i][..., 1], min_distance=min_dist_peaks, indices=True,
                                                     threshold_rel=0.2, num_peaks=nber_peaks)
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

    if len(tmps) != 0:
        idx, count = np.unique(np.concatenate(tmps), return_counts=True)
        # In case of multiple peaks counted at least twice, go to intensity and pick the peak with highest intensity
        conc_all_peaks = conc_all_peaks[idx]
        conc_all_peaks_intensities = conc_all_peaks_intensities[idx]

    x, y = conc_all_peaks[np.argmax(conc_all_peaks_intensities)]

    return x, image.shape[0]-y

def HatShirtBankRW(shirt_stripe_thickness=[3, 4, 5, 6], shirt_stripe_nber=6, distance_hat_shirt=1.5, angle=45):
    bank = []
    for thick in shirt_stripe_thickness:
        bank.append(HatShirtMotifRW(thick, shirt_stripe_nber, distance_hat_shirt, angle))
    return bank