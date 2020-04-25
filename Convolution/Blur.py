import cv2
import numpy as np
img = cv2.imread('lena_color_256.tif',1)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rows,cols = img.shape[:2]
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#blur
kernel_3x3 = np.ones((3,3),np.float32) / 9.0
output = cv2.filter2D(img,-1,kernel_3x3)
cv2.imshow('Identity filter',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#larger blur
kernel_5x5 = np.ones((5,5),np.float32) / 25.0
output=cv2.filter2D(img,-1,kernel_5x5)
cv2.imshow('Identity filter',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#motion blur
size=15
kernel_motion_blur = np.zeros((size,size))
kernel_motion_blur[int((size-1)/2),:] = np.ones(size)
kernel_motion_blur = kernel_motion_blur /  size
output=cv2.filter2D(img,-1,kernel_motion_blur)
cv2.imshow('Motion blur',output)
cv2.waitKey(0)
cv2.destroyAllWindows()