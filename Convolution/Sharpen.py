import cv2
import numpy as np
img = cv2.imread('lena_color_256.tif',1)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rows,cols = img.shape[:2]
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#sharpen
kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
output1=cv2.filter2D(img,-1,kernel_sharpen_1)
cv2.imshow('Sharpening',output1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#more shapening
kernel_sharpen_2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
output2=cv2.filter2D(img,-1,kernel_sharpen_2)
cv2.imshow('Excessive sharpening',output2)
cv2.waitKey(0)
cv2.destroyAllWindows()