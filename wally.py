from pylab import imshow
import numpy as np
import mahotas  # pip install mahotas

wally = mahotas.imread('./data/images/02.jpg')

wfloat = wally.astype(float)
r, g, b = wfloat.transpose((2, 0, 1))

w = wfloat.mean(2)

pattern = np.zeros((25, 25), float)
for i in range(5):
    pattern[i::10] = 255

v = mahotas.convolve(r - w, pattern)

mask = (v == v.max())
mask = mahotas.dilate(mask, np.ones((48, 24)))

np.subtract(wally, .8*wally * ~mask[:, :, None], out=wally, casting='unsafe')
imshow(wally)


