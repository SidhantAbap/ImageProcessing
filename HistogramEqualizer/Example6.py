import cv2
import matplotlib.pyplot as plt
import numpy as np

#Read the image in grey scale
img = cv2.imread('cameraman.tif',0)

#CV2 equalization option1
output1 = cv2.equalizeHist(img)

#CV2 equalization option2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
output2 = clahe.apply(img)

output = [img, output1, output2]
titles = ['Orginal Image', 'K=2', 'K=4']

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(output[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()