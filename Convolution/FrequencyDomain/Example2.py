import cv2
import numpy as np
import matplotlib.pyplot as plt
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

img = cv2.imread('lena_color_256.tif',0)

#Gaussian blur
kernel_3x3 = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.0
output = cv2.filter2D(img,-1,kernel_3x3)

#Magnitude spectrum original image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

#Magnitude spectrum of blured image
f1 = np.fft.fft2(output)
f1shift = np.fft.fftshift(f1)
magnitude_spectrum1 = 20*np.log(np.abs(f1shift))

#Inverse FFT on magnitude spectrum blur image
LowPassCenter = f1shift * idealFilterLP(50,output.shape)
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
plt.imshow(magnitude_spectrum1)
plt.title('Magnitude Spectrum blur', size=20)

plt.subplot(2,3,4)
plt.imshow(Img_recon)
plt.title('Recon Image', size=20)
plt.show()