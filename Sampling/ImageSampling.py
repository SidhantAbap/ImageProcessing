import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
im = cv2.imread('umbc.png',0)
print(im.shape)
plt.figure(figsize=(5,5))
plt.imshow(im, cmap= 'gray')
plt.show()

#Apply gausian blur
plt.figure(figsize=(5,5))
im_blurred = gaussian_filter(im, sigma=2.5) #(5,5,1)
plt.imshow(im_blurred, cmap= 'gray')
plt.show()

#Down sampling without bluring
n = 4 # create and image 16 times smaller in size
w, h = im.shape[0] // n, im.shape[1] // n
im_small = np.zeros((w,h))
for i in range(w):
   for j in range(h):
      im_small[i,j] = im[n*i, n*j]
plt.figure(figsize=(5,5))
plt.imshow(im_small, cmap='gray')
plt.show()

#Down Sampling with gausian blur
im_small = np.zeros((w,h))
for i in range(w):
   for j in range(h):
      im_small[i,j] = im_blurred[n*i, n*j]
plt.figure(figsize=(5,5))
plt.imshow(im_small, cmap='gray')
plt.show()