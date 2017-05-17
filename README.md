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




