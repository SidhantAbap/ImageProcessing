import cv2
import numpy as np
img = cv2.imread('lena_color_256.tif',1)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rows,cols = img.shape[:2]
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#identity
kernel_identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
output = cv2.filter2D(img,-1,kernel_identity)
cv2.imshow('Identity filter',output)
cv2.waitKey(0)
cv2.destroyAllWindows()