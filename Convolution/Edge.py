import cv2
import numpy as np
img = cv2.imread('lena_color_256.tif',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows,cols = img.shape[:2]
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#edge enhancement
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
output3=cv2.filter2D(img,-1,kernel_sharpen_3)
cv2.imshow('Edge Enhancement',output3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#emboss
kernel_emboss1=np.array([[0,-1,1],[1,0,-1],[1,1,0]])
output=cv2.filter2D(gray,-1,kernel_emboss1)+128
cv2.imshow('Emboss1',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#emboss2
kernel_emboss2=np.array([[-1,-1,0],[-1,0,11],[0,1,1]])
output=cv2.filter2D(gray,-1,kernel_emboss2)+128
cv2.imshow('Emboss2',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#emboss3
kernel_emboss3=np.array([[1,0,0],[0,0,0],[0,0,-1]])
output=cv2.filter2D(gray,-1,kernel_emboss3)+128
cv2.imshow('Emboss3',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Sobel
sobel_horizontal = cv2.Sobel(img,cv2.CV_64F, 1,0,ksize=5)
cv2.imshow('Sobel horizontal',sobel_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Sobel2
sobel_vertical = cv2.Sobel(img,cv2.CV_64F, 1,0,ksize=5)
cv2.imshow('Sobel vertical',sobel_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()

#erode
kernel_erode=np.ones((5,5),np.uint8)
img_erosion = cv2.erode(img,kernel_erode,iterations = 1)
cv2.imshow('Erode',img_erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#dilate
img_dilatation = cv2.dilate(img,kernel_erode,iterations = 1)
cv2.imshow('Dilate',img_dilatation)
cv2.waitKey(0)
cv2.destroyAllWindows()

#vignette
kernel_gauss_x= cv2.getGaussianKernel(cols,200)
kernel_gauss_y = cv2.getGaussianKernel(rows,200)
kernel = kernel_gauss_y * kernel_gauss_x.T
mask=255*kernel/np.linalg.norm(kernel)
output=np.copy(img)
for i in range(3):
  output[:,:,i]=output[:,:,i] * mask
cv2.imshow('Vignette',output)
cv2.waitKey(0)
cv2.destroyAllWindows()
