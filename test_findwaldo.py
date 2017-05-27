from findwaldo import find_waldo
import matplotlib.pyplot as plt
import numpy as np

image = plt.imread('./data/images/06.jpg')

x, y = find_waldo(image)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

image_grey = rgb2gray(image)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
plt.imshow(image_grey, cmap='gray')
rect = plt.Rectangle((x-15, image.shape[0]-y-15), 30, 30, edgecolor='r', facecolor='none')
ax1.add_patch(rect)

