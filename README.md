# ImageProcessing_TeamProject
Team project to lecture "Introduction to Signal and Image Processing" at University of Bern.

### Team members
- Livio Baetscher
- Manueal Haefliger
- Marc-Antoine Jacques

### Language
- Python 3.6.1

### Considered Approaches

Both of the first approaches were based on template matching by convolution/correlation using **grayscale images**.

**1. Detect Waldo's striped shirt.**
- Template: use a generic image representative of Waldo's shirt or a pattern generated "in silico".
- Preprocessing: extract only red pixels from the image and binarize.
- Pros: efficient reduction of the background before performing the convolution.
- Cons: Waldo's shirt is **not always present** in the images, stripe patterns are also introduced here and there in the
images. Need to specify the RGB values to use for extracting red pixels -> "hard thresholds"

**2. Detect Waldo's face.**
- Template: use a generic image of Waldo's face.
- Preprocessing: (edge detection?)
- Pros: head should always be present
- Cons: What size for the template? (note this is a common problem for convolution-like methods, workaround is to create
other versions of the image by applying a Gaussian pyramid transformation)

**3. Detect Waldo's glasses.**

I don't see any clear advantages over face detection here, but circle detections could be used as a final touch for both 
previous methods when hits are ambiguous.

**4. Machine learning approach (MLP, deep neural network...)**

Sounds sexy and extremely robust but a lot of work to: get training libraries and would need a robust segmentation of
images to identify the characters in them.


### How to perform the template matching well

Nice links:
http://www.cse.psu.edu/~rtc12/CSE486/lecture07.pdf

https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.correlate2d.html (watch closely the example)

- Use convolution in the Fourier's space (fftconvolve) we perform exactly the same operation as regular convolution but
much faster.
- Don't forget that convolution reverses the template it matches, so flip the template before performing the convolution
to obtain a "correlation-like" operation.
- Usually people use normalized cross-correlation (NCC) instead of convolution. The effect of these operations is
extremely similar but the difference is that **NCC corrects locally the "patches" (i.e. the template and the part of the 
image with which it is applied) for intensity mean and deviation**. NCC is not implemented in skimage but it seems that
simply removing the mean intensity of the image from the image and the mean intensity of the template from the template
BEFORE performing the convolution gives us a nice workaround.



### New development: filters bank

Because neither head only or stripes only matching seems to convince, we want to build a bank of filters, with the idea
that we could then combine the heatmaps of these filters and return hottest area as most probably position of Waldo.

I've been through all images and have extracted some features from Waldo: see ./template/features.tsv  the idea is to 
use these filters to know their dimensions and characteristics

The library would comprise 2 different types of filters: stripes and head. How to define these filters:
- stripes filters are defined by the number of stripes and height of stripes. It seems that 4 stripes look reasonable, 
with an equal height for red or white stripes. Total height of the filter should vary between 12 and 28 pixels (i.e.
stripes between 3 and 7 pixel high).

- faces: if you go through the images it appears that faces can be pretty different from one picture to another, but it
seems that they are fixed models that are then reused. But how many and which? For that I've just measured the dimensions
of the faces, do some plotting around see: analysis_head_features.R. In the end I would just create 2 groups of faces 
one for "regular" (ratio height/width = 1.8) and one for "long" faces (ratio height/width = 2.3). With 4 sizes (height
without hair): 5, 12, 15, 30 px.  



 
 #### Preprocessing and hit refinement
 
 I would keep the extraction of red pixels when it comes to stripe detection. To select what is a red pixel a better 
 idea would be to use what humans see as red head of fixed thresholhd in RGB space.
 
 For face detection I would work in grayscale space. I'm not sure, but maybe a Canny edge detector could help as well.
 
 A final hit refinement could look for circles that would fit Waldo's glasses.