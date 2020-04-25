from PIL import Image
from skimage.io import imread, imshow, show
import scipy.fftpack as fp
from scipy import ndimage, misc, signal
#from scipy.stats import signaltonoise
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.transform import rescale
import matplotlib.pylab as pylab
import numpy as np
import numpy.fft
import timeit
import cv2

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

#Import the image
img = cv2.imread('lena_color_256.tif',0)

#Create a Gaussian kernel
kernel = np.outer(signal.gaussian(img.shape[0], 5), signal.gaussian(img.shape[1], 5))
#print(kernel.shape)

#convert digital image to frequency domain
#Fast Fourier transform
freq = fp.fft2(img)

#Error if the shape of both image and kernel not equal
assert(freq.shape == kernel.shape)

#Convert the kernle into frequency domain
freq_kernel = fp.fft2(fp.ifftshift(kernel))
#print(freq.shape)
#print(freq_kernel.shape)

#Convolve the frequency domain image and kernel
convolved = freq*freq_kernel
#print(convolved.shape)

#Inverse FFT to get the real image from frequency domain
im1 = fp.ifft2(convolved).real

pylab.figure(figsize=(20,15))
pylab.gray() # show the filtered result in grayscale
pylab.subplot(2,3,1), pylab.imshow(img), pylab.title('Original Image',
size=20), pylab.axis('off')
pylab.subplot(2,3,2), pylab.imshow(kernel), pylab.title('Gaussian Kernel', size=20)
pylab.subplot(2,3,3), pylab.imshow(im1) # the imaginary part is an artifact
pylab.title('Output Image', size=20), pylab.axis('off')
pylab.subplot(2,3,4), pylab.imshow( (20*np.log10( 0.1 + fp.fftshift(freq))).astype(int))
pylab.title('Original Image Spectrum', size=20), pylab.axis('off')
pylab.subplot(2,3,5), pylab.imshow( (20*np.log10( 0.1 +
fp.fftshift(freq_kernel))).astype(int))
pylab.title('Gaussian Kernel Spectrum', size=20), pylab.subplot(2,3,6)
pylab.imshow( (20*np.log10( 0.1 + fp.fftshift(convolved))).astype(int))
pylab.title('Output Image Spectrum', size=20), pylab.axis('off')
pylab.subplots_adjust(wspace=0.2, hspace=0)
pylab.show()