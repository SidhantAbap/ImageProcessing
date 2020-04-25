import cv2
import numpy as np
img = cv2.imread('cameraman.tif',0)
edge_laplace_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
im_edges = cv2.filter2D(img, -1,edge_laplace_kernel)
cv2.imshow('Laplacian Kernel',im_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()