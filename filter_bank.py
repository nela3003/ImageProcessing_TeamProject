from preprocessing_utils import *
from skimage.feature import match_template

def StripeBank(heights=[12, 16, 24, 28], width=2, nber_stripes=4):
    bank = []
    for height in heights:
        bank.append(StripeMotif(height, width, nber_stripes))
    return bank


def HatShirtBank(shirt_stripe_thickness=[3, 4, 5 , 6], shirt_stripe_nber=6, distance_hat_shirt=1.5, angle=45):
    bank = []
    for thick in shirt_stripe_thickness:
        bank.append(HatShirtMotif(thick, shirt_stripe_nber, distance_hat_shirt, angle))
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

def HeadBank(px = [5, 12, 15, 30]):
    from scipy.misc import imresize
    bank = []
    long = plt.imread('./template/head/05.png')
    long = rgb2gray(long)
    wide = plt.imread('./template/head/27.png')
    wide = rgb2gray(wide)
    for p in px:
        tmplt_wide = imresize(wide, (int(wide.shape[0] / wide.shape[1] * p), p),
                         interp='bilinear', mode=None)
        tmplt_long = imresize(long, (int(long.shape[0] / long.shape[1] * p), p),
                         interp='bilinear', mode=None)
        bank.append(tmplt_long)
        bank.append(tmplt_wide)
    return bank


stripes_bank = StripeBank(heights=[18, 24, 30, 36], width=4, nber_stripes=6)
glasses_bank = GlassesBank()
heads_bank = HeadBank()
hatshirt_bank = HatShirtBank()

total_bank = stripes_bank + glasses_bank + heads_bank + hatshirt_bank


######### FFT Version #####################
t=time.time()
image = './data/images/04.jpg'
image = plt.imread(image)
grayscale = rgb2gray(image)
reds = ExtractRed(image, 150, 100, 100)
grayscale_reds = rgb2gray(reds)
blacks = ExtractBlack(image, 150, 150, 150)
grayscale_blacks = rgb2gray(blacks)
grayscale = rgb2gray(image)
grayscale -= np.mean(grayscale)

response = []
i = 0 #  First filters are stripes
for filt in total_bank:
    template = filt.copy()
    template = np.fliplr(template)
    template = np.flipud(template)
    template -= int(np.mean(template))
    if i <= 3:  # stripes
        score = scipy.signal.fftconvolve(grayscale_reds, template, mode='same')
    elif i > 3 and i <= 12:  # glasses
        score = scipy.signal.fftconvolve(grayscale_blacks, template, mode='same')
    else:  # heads
        score = scipy.signal.fftconvolve(grayscale, template, mode='same')
    score = (score - score.mean())/score.std()
    response.append(score)
    i += 1

print('elpased time: {:02f}'.format(time.time() - t))
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

fig = plt.figure()
for i in range(len(hatshirt_bank)):
    plt.subplot(2,2,i+1)
    plt.imshow(hatshirt_bank[i])
plt.show()



#################### RW Image ######################
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


def HatShirtBankRW(shirt_stripe_thickness=[3, 4, 5 , 6], shirt_stripe_nber=6, distance_hat_shirt=1.5, angle=45):
    bank = []
    for thick in shirt_stripe_thickness:
        bank.append(HatShirtMotifRW(thick, shirt_stripe_nber, distance_hat_shirt, angle))
    return bank


hatshirtRW_bank = HatShirtBankRW(shirt_stripe_nber=4)
fig = plt.figure()
for i in range(4):
    plt.subplot(2,4,i+1)
    plt.imshow(hatshirtRW_bank[i][..., 0])
for i in range(4):
    plt.subplot(2,4,i+5)
    plt.imshow(hatshirtRW_bank[i][..., 1])
plt.show()

t=time.time()
image = './data/images/03.jpg'
image = plt.imread(image)
reds = ExtractRed(image, 150, 100, 100)
grayscale_reds = rgb2gray(reds)
grayscale_reds -= np.mean(grayscale_reds)
whites = ExtractWhite(image, 220, 220, 180)
grayscale_whites = rgb2gray(whites)
grayscale_whites -= np.mean(grayscale_whites)
rw_image = np.empty((image.shape[0], image.shape[1], 2))
rw_image[..., 0] = grayscale_reds
rw_image[..., 1] = grayscale_whites

response = []
for filt in hatshirtRW_bank:
    template = filt.copy()
    template = np.fliplr(template)
    template = np.flipud(template)
    template -= int(np.mean(template))
    score = scipy.signal.fftconvolve(rw_image, template, mode='same')
    #score = (score - score.mean())/score.std()
    response.append(score)

print('elpased time: {:02f}'.format(time.time() - t))

# Collapse RW response
plt.figure()
combined_response = []
nb_plot = 1
for i in range(len(response)):
    #combined_response.append(np.sum(response[i], axis=2))
    #peak_positions = feature.corner_peaks(combined_response[i], min_distance=200, indices=True, threshold_rel=0.2, num_peaks=5)
    #peak_positions_img = feature.corner_peaks(combined_response[i], min_distance=200, indices=False, threshold_rel=0.2, num_peaks=5)
    reds_peak_positions = feature.corner_peaks(response[i][..., 0], min_distance=200, indices=True, threshold_rel=0.2, num_peaks=5)
    reds_peak_positions_img = feature.corner_peaks(response[i][..., 0], min_distance=200, indices=False, threshold_rel=0.2, num_peaks=5)
    whites_peak_positions = feature.corner_peaks(response[i][..., 1], min_distance=200, indices=True, threshold_rel=0.2, num_peaks=5)
    whites_peak_positions_img = feature.corner_peaks(response[i][..., 1], min_distance=200, indices=False, threshold_rel=0.2, num_peaks=5)
    for pos in reds_peak_positions:
        DrawRectangle(reds_peak_positions_img, pos[0], pos[1], 15)
    plt.subplot(2, 4, nb_plot)
    nb_plot += 1
    plt.imshow(image)
    plt.imshow(reds_peak_positions_img, alpha=0.7)
    plt.title('Red Filter: '+str(i))
    for pos in whites_peak_positions:
        DrawRectangle(whites_peak_positions_img, pos[0], pos[1], 15)
    plt.subplot(2, 4 ,nb_plot)
    nb_plot += 1
    plt.imshow(image)
    plt.imshow(whites_peak_positions_img, alpha=0.7)
    plt.title('White Filter: '+str(i))
plt.show()

# Detect close peaks
#for i in range(len(response)):
#    combined_response.append(np.sum(response[i], axis=2))