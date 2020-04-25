import cv2
import matplotlib.pyplot as plt
import numpy as np

#Read the image in grey scale
img = cv2.imread('peppers.png',1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

red_hist = cv2.calcHist([img], [0], None, [256], [0,255])
green_hist = cv2.calcHist([img], [1], None, [256], [0,255])
blue_hist = cv2.calcHist([img], [2], None, [256], [0,255])

plt.subplot(2,2,1)
plt.imshow(img,)
plt.title('Image', size=20)
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.xlim([0,255])
plt.plot(red_hist)
plt.title("Red Histogram")

plt.subplot(2,2,3)
plt.xlim([0,255])
plt.plot(green_hist)
plt.title("Green Histogram")

plt.subplot(2,2,4)
plt.xlim([0,255])
plt.plot(blue_hist)
plt.title("Blue Histogram")
plt.show()