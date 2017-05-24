from preprocessing_utils import *

def StripeBank(heights=[12, 16, 24, 28], width=2, nber_stripes=4):
    bank = []
    for height in heights:
        bank.append(StripeMotif(height, width, nber_stripes))
    return bank

stripes_bank = StripeBank(width=5)



######### FFT Version #####################
image = './data/images/04.jpg'
image = plt.imread(image)
reds = ExtractRed(image, 150, 100, 100)
grayscale = rgb2gray(reds)
grayscale -= np.mean(grayscale)

response = []
for filt in stripes_bank:
    template = filt.copy()
    template = np.fliplr(template)
    template = np.flipud(template)
    template -= np.mean(template)
    score = scipy.signal.fftconvolve(grayscale, template, mode='same')
    response.append(score)

temp = sum(response)
peak_positions = feature.corner_peaks(temp, min_distance=200, indices=True, threshold_rel=0.2, num_peaks=5)
peak_positions_img = feature.corner_peaks(temp, min_distance=200, indices=False, threshold_rel=0.2, num_peaks=5)
for pos in peak_positions:
    DrawRectangle(peak_positions_img, pos[0], pos[1], 15)
PlotHeatmap(temp, peak_positions_img, title='Most probable positions of Waldo', bar=False)



############### Open CV CCOR #########
"""
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
"""

image = './data/images/04.jpg'
image = plt.imread(image)
image_intact = image.copy()
image = ExtractRed(image, 150, 100, 100)
image = rgb2gray(image)
img2 = image.copy()
#img2 -= np.mean(img2)
img2 = img2.astype('float32')  # Float32 for opencv compatibility


response = []
max_score = 0
best_loc = (0,0)
best_filt = None
i=0
for filt in stripes_bank:
    template = filt.copy()
    template -= np.mean(template)
    template = template.astype('float32')  # Float32 for opencv compatibility
    method = eval('cv2.TM_CCORR_NORMED')
    score = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score)
    if max_val > max_score:
        best_loc = min_loc
        max_score = min_val
        best_filt = i

        # Use for drawing rectangle where it matched (hard to spot, looks like a thick red vertical line)
        top_left = min_loc
        w, h = template.shape[::-1]
        bottom_right = (top_left[0] + w, top_left[1] + h)
    i += 1
    response.append(score)

cv2.rectangle(image_intact, top_left, bottom_right, 255, 10)
plt.figure()
plt.subplot(121), plt.imshow(response[best_filt], cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image_intact, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()