import cv2
import matplotlib.pyplot as plt

#Read the image in grey scale
img = cv2.imread('peppers.png',1)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
R, G, B = cv2.split(img)

plt.subplot(2,2,1)
plt.imshow(img,)
plt.title('Image', size=20)
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.hist(R.ravel(), 256, [0,255])
plt.title('Red Channel Histogram', size=20)
plt.xlim(xmin= 0, xmax=256)

plt.subplot(2,2,3)
plt.hist(G.ravel(), 256, [0,255])
plt.title('Green Channel Histogram', size=20)
plt.xlim(xmin= 0, xmax=256)

plt.subplot(2,2,4)
plt.hist(B.ravel(), 256, [0,255])
plt.title('Blue Channel Histogram', size=20)
plt.xlim(xmin= 0, xmax=256)
plt.show()