import cv2
import numpy as np
img = cv2.imread('lena_color_256.tif',1)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rows,cols = img.shape[:2]
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Gaussian blur
kernel_3x3 = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.0
output = cv2.filter2D(img,-1,kernel_3x3)
cv2.imshow('Gaussian filter',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#add some gaussian noise
img = cv2.imread('lena_color_256.tif',0)
noise_gaussian = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
cv2.randn(noise_gaussian, 64, 32)
noisy_image = cv2.add(img, noise_gaussian)
cv2.imshow("Gaussian noise added - severe", noisy_image)
cv2.waitKey()

#Gaussian Filter on noisy image
output = cv2.filter2D(img,-1,kernel_3x3)
cv2.imshow('Gaussian filter Noisy',output)
cv2.waitKey(0)
cv2.destroyAllWindows()