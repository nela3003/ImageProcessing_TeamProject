import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
import scipy.misc

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# ---------------------------------------------------------------------------
# GLASSES

k = '05'

px = [10, 12, 14, 16, 18, 20, 22, 24, 26]

result_all = list()

wally = plt.imread('./data/images/'+k+'.jpg')
template = plt.imread('./template/glasses.png')

wally_grey = rgb2gray(wally)
template_grey = rgb2gray(template)

for p in px:
    tmplt = scipy.misc.imresize(template_grey, (int(template_grey.shape[0] / template_grey.shape[1] * p), p),
                                        interp='bilinear', mode=None)
    result = match_template(wally_grey, tmplt)
    result_all.append(result)


best = [np.max(r) for r in result_all]

idx = np.argmax(best)

ij = np.unravel_index(np.argmax(result_all[idx]), result_all[idx].shape)

x, y = ij[::-1]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
plt.imshow(wally_grey, cmap='gray')
rect = plt.Rectangle((x, y), 30, 30, edgecolor='r', facecolor='none')
ax1.add_patch(rect)


# ---------------------------------------------------------------------------
# HEAD


k = '30'

hd = ['01', '02', '03', '04', '05', '06', '07', '09', '10', '11', '12', '13', '15',
      '17', '18', '21', '22', '25', '26', '27']

result_all = list()

wally = plt.imread('./data/images/'+k+'.jpg')


wally_grey = rgb2gray(wally)

# for p in px:
#     tmplt = scipy.misc.imresize(template_grey, (int(template_grey.shape[0] / template_grey.shape[1] * p), p),
#                                         interp='bilinear', mode=None)
#     result = match_template(wally_grey, tmplt)
#     result_all.append(result)

for h in hd:
    template = plt.imread('./template/head/' + h + '.png')
    template_grey = rgb2gray(template)
    result = match_template(wally_grey, template_grey)
    result_all.append(result)

best = [np.max(r) for r in result_all]

idx = np.argmax(best)

ij = np.unravel_index(np.argmax(result_all[idx]), result_all[idx].shape)

x, y = ij[::-1]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
plt.imshow(wally_grey, cmap='gray')
rect = plt.Rectangle((x, y), 30, 30, edgecolor='r', facecolor='none')
ax1.add_patch(rect)