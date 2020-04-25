from PIL import Image
from collections import defaultdict
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('hist2.tif',0)
original_image = image
cv2.imshow('Original Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plotting Histogram
def histogram(image,text):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.title(text)
    plt.show()
# Original Image Histogram
histogram(original_image,"Original Image")

#Algorithm
a = np.zeros((256,),dtype=np.float16)
b = np.zeros((256,),dtype=np.float16)

height,width=image.shape

#finding histogram
for i in range(width):
    for j in range(height):
        g = image[j,i]
        a[g] = a[g]+1

print(a)

#performing histogram equalization
tmp = 1.0/(height*width)
b = np.zeros((256,),dtype=np.float16)

for i in range(256):
    for j in range(i+1):
        b[i] += a[j] * tmp;
    b[i] = round(b[i] * 255);

# b now contains the equalized histogram
b=b.astype(np.uint8)

#Re-map values from equalized histogram into the image
for i in range(width):
    for j in range(height):
        g = image[j,i]
        image[j,i]= b[g]

cv2.imshow('Equalized Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()