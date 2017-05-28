from findwaldo import find_waldo
import matplotlib.pyplot as plt
import numpy as np

# remember: 14, 16, 19, 24 don't exist
image = plt.imread('./data/images/06.jpg')

x, y = find_waldo(image)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

image_grey = rgb2gray(image)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
# show the original image in grayscale that the rectangle is better visible
plt.imshow(image_grey, cmap='gray')
# draw a red rectangle of size 30x30 around the pixel
rect = plt.Rectangle((x-15, image.shape[0]-y-15), 30, 30, edgecolor='r', facecolor='none')
ax1.add_patch(rect)

# with nber_peaks = 1 and min_dist_peaks = 100
# found: 04, 06, 10, 12, 21, 27

# with nber_peaks = 1 and min_dist_peaks = 150
# found: 04, 06, 12, 21, 27

# with nber_peaks = 3 and min_dist_peaks = 100
# found: 04, 06, 12, 21, 27

# with nber_peaks = 1 and min_dist_peaks = 100 and glasses
# found: 04, 06, 10, 12, 21, 27



########################################################################


import os
import time
import matplotlib.pyplot as plt

files = ['./data/images/' + i  for i in os.listdir('./data/images/')]
t0 = time.time()
result = []
for file in files:
    print(file)
    img = plt.imread(file)
    result.append(find_waldo(img))
t1 = time.time()
print('Elapsed time: {:02f}'.format(t1 - t0))

with open('temp.txt', 'w') as f:
    for res in result:
        f.write(str(res) + '\n')