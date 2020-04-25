import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('hist2.tif',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)

#hist,bins = np.histogram(img.flatten(),256,[0,256])
#plt.hist(img.flatten(),256,[0,256], color = 'r')
#plt.xlim([0,256])
#plt.show()

#Pltting figure size
plt.figure(figsize=(15,10))
plt.gray() # show the filtered result in grayscale
plt.subplot(2,3,1)
plt.imshow(img)
plt.title('Original Image', size=20)

plt.subplot(2,3,2)
plt.imshow(equ)
plt.title('Enhanced Image', size=20)
plt.show()