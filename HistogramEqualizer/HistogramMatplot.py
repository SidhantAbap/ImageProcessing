import cv2
import matplotlib.pyplot as plt

#Read the image in grey scale
img = cv2.imread('peppers_BL.tif',0)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Image', size=20)
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.hist(img.ravel(), 256, [0,255])
plt.title('Histogram', size=20)
plt.xlim(xmin= 0, xmax=256)
plt.show()

