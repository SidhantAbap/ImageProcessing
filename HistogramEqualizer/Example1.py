import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('hist2.tif',0)

#PDF
hist,bins = np.histogram(img.flatten(),256,[0,256])

#CDF
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]

plt.plot(cdf_m, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

#Pltting figure size
plt.figure(figsize=(15,10))
plt.gray() # show the filtered result in grayscale
plt.subplot(2,3,1)
plt.imshow(img)
plt.title('Original Image', size=20)

plt.subplot(2,3,2)
plt.imshow(img2)
plt.title('Enhanced Image', size=20)
plt.show()