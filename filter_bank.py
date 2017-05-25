from preprocessing_utils import *
from skimage.feature import match_template

def StripeBank(heights=[12, 16, 24, 28], width=2, nber_stripes=4):
    bank = []
    for height in heights:
        bank.append(StripeMotif(height, width, nber_stripes))
    return bank

def GlassesBank(px = [10, 12, 14, 16, 18, 20, 22, 24, 26]):
    from scipy.misc import imresize
    bank = []
    template = plt.imread('./template/glasses.png')
    template_grey = rgb2gray(template)
    for p in px:
        tmplt = imresize(template_grey, (int(template_grey.shape[0] / template_grey.shape[1] * p), p),
                                interp='bilinear', mode=None)
        bank.append(tmplt)
    return bank

stripes_bank = StripeBank(heights=[18, 24, 30, 36], width=2, nber_stripes=6)
glasses_bank = GlassesBank()

total_bank = stripes_bank + glasses_bank


######### FFT Version #####################
image = './data/images/09.jpg'
image = plt.imread(image)
reds = ExtractRed(image, 150, 100, 100)
grayscale_reds = rgb2gray(reds)
blacks = ExtractBlack(image, 150, 150, 150)
grayscale_blacks = rgb2gray(blacks)
grayscale = rgb2gray(image)
grayscale -= np.mean(grayscale)

response = []
i = 0 #  First filters are stripes
for filt in stripes_bank:
    template = filt.copy()
    template = np.fliplr(template)
    template = np.flipud(template)
    template -= int(np.mean(template))
    if i <= 3:  # stripes
        score = scipy.signal.fftconvolve(grayscale_reds, template, mode='same')
    else:  # glasses
        score = scipy.signal.fftconvolve(grayscale_blacks, template, mode='same')
    score = (score - score.mean())/score.std()
    response.append(score)
    i += 1

temp = sum(response)
peak_positions = feature.corner_peaks(temp, min_distance=200, indices=True, threshold_rel=0.2, num_peaks=1)
peak_positions_img = feature.corner_peaks(temp, min_distance=200, indices=False, threshold_rel=0.2, num_peaks=1)
for pos in peak_positions:
    DrawRectangle(peak_positions_img, pos[0], pos[1], 15)
PlotHeatmap(temp, peak_positions_img, title='Most probable positions of Waldo', bar=True)
PlotHeatmap(image, temp)


############### Open CV CCOR #########
"""
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
"""

image = './data/images/27.jpg'
image = plt.imread(image)
image_intact = image.copy()
image = ExtractRed(image, 150, 100, 100)
image = rgb2gray(image)
img2 = image.copy()
#img2 -= np.mean(img2)

response = []
max_score = 0
best_loc = (0,0)
best_filt = None
i=0
for filt in stripes_bank:
    template = filt.copy()
    template -= np.mean(template)
    score = match_template(img2, template, pad_input=True)
    max_val = score.max()
    if max_val > max_score:
        best_loc = np.unravel_index(score.argmax(), score.shape)
        max_score = max_val
        best_filt = i
        # Use for drawing rectangle where it matched (hard to spot, looks like a thick red vertical line)
    i += 1
    response.append(score)

DrawRectangle(image_intact, best_loc[0], best_loc[1], 10)
plt.imshow(image_intact)
PlotHeatmap(image, score[best_filt])