import numpy as np
import numpy.fft as fp
import matplotlib.pyplot as plt
import cv2
from math import sqrt,exp

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

#Low pass filter
def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

#Read the image in grey scale
img = cv2.imread('peppers_BL.tif',0)
print(img.shape)

#Prepare a mask for zero padding/sampling
mask = np.zeros_like(img)
mask[::2, 0::2] = True
mask[::1, 1::2] = False
print(mask.shape)

#Np arrya with zeros of origional image size
im1 = np.zeros((img.shape[0], img.shape[1]))

#Fill the sampled image array
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i,j] == 1:
            im1[i,j] = img[i,j]
        else:
            im1[i, j] = 0
#image = cv2.dft(np.float32(im1), flags= cv2.DFT_COMPLEX_OUTPUT)

#Magnitude spectrum original image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

#Magnitude spectrum sampled image
Yfft = np.fft.fft2(im1)
Y = np.fft.fftshift(Yfft)
magnitude_spectrum2 = 20*np.log(np.abs(Y))

LowPassCenter = Y * idealFilterLP(50,im1.shape)
magnitude_spectrum3 = np.log(1+np.abs(LowPassCenter))

LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
Img_recon = np.abs(inverse_LowPass)

#Pltting figure size
plt.figure(figsize=(15,10))
plt.gray() # show the filtered result in grayscale

plt.subplot(2,3,1)
plt.imshow(img)
plt.title('Original Image', size=20)

plt.subplot(2,3,2)
plt.imshow(magnitude_spectrum)
plt.title('Magnitude Spectrum', size=20)

plt.subplot(2,3,3)
plt.imshow(im1)
plt.title('Padded Image', size=20)

plt.subplot(2,3,4)
plt.imshow(magnitude_spectrum2)
plt.title('Magnitude Spectrum2', size=20)

plt.subplot(2,3,5)
plt.imshow(Img_recon)
plt.title('Recon Image', size=20)

plt.subplot(2,3,6)
plt.imshow(magnitude_spectrum3)
plt.title('Magnitude Spectrum3', size=20)
plt.show()