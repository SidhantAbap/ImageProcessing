import cv2
img = cv2.imread('cameraman.tif', 0)
print(img.shape)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

near_img = cv2.resize(img,(1024,1024), interpolation = cv2.INTER_NEAREST)
print(near_img.shape)
cv2.imshow('image',near_img)
cv2.waitKey(0)
#cv2.destroyAllWindows()

bilinear_img = cv2.resize(img,None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
print(bilinear_img.shape)
cv2.imshow('image2',bilinear_img)
cv2.waitKey(0)
#cv2.destroyAllWindows()

bicubic_img = cv2.resize(img,None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
print(bicubic_img.shape)
cv2.imshow('image3',bicubic_img)
cv2.waitKey(0)
#cv2.destroyAllWindows()