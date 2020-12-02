import numpy as np
import cv2
from scipy import signal
import dippykit as dip

img = cv2.imread("test.png")
if img.ndim > 2:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blkRadii = 7
blkSize = 2 * blkRadii +1
npts = 20
r = img.shape[0]
c = img.shape[1]
effr = r - blkSize
effc = c - blkSize


xx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
yy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
Gx = signal.convolve2d(img,xx,boundary='symm',mode='same')    
Gy = signal.convolve2d(img,yy,boundary='symm',mode='same')    
Igradient_fd = 5
featureimg_total = np.zeros((r,c,Igradient_fd))
featureimg_total[:,:,0] = img
featureimg_total[:,:,1] = Gx
featureimg_total[:,:,2] = np.abs(Gx)
featureimg_total[:,:,3] = Gy
featureimg_total[:,:,4] = np.abs(Gy)

Igradient_fd = 5
for j in np.arange(Igradient_fd):
    featureimg = featureimg_total[:,:,j]
    dfeatureimg = featureimg.astype('double')
    itimg = np.cumsum(featureimg,axis=0)
    # dip.imshow(itimg,'gray')
    # dip.show()
    # cv2.imshow('imshow2',itimg/8)
    itimg = np.cumsum(itimg,axis=1)
    iimg = np.zeros((itimg.shape[0]+2,itimg.shape[1]+2))
    iimg[1:-1,1:-1] = itimg
    dip.imshow(iimg,'gray')
    dip.show()
    # dip.imshow(featureimg_total[:,:,j],'gray')
    # dip.show()
     # cv2.imshow('imshow1',itimg/8)
    # cv2.imshow('imshow0',featureimg_total[:,:,j]/8)

    cv2.waitKey(0)
cv2.destroyAllWindows()