import numpy as np
import cv2
# from skimage import data,filters,img_as_ubyte
from scipy import signal

def computeIGradientDmd(img, dmdOpts):
    if img.ndim > 2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    pts1 = dmdOpts['xi'] 
    pts2 = dmdOpts['yi']
    blkRadii = dmdOpts['radii']
    gridSpacing = dmdOpts['gridspace'] 
    maxScale = dmdOpts['scale']
    numSamp = pts1.shape[1]
    samp_per_scale = numSamp / maxScale

    pts1[pts1 > blkRadii - maxScale + 1] = blkRadii - maxScale + 1
    pts1[pts1 < -blkRadii] = -blkRadii
    pts2[pts2 > blkRadii - maxScale + 1] = blkRadii - maxScale + 1
    pts2[pts2 < -blkRadii] = -blkRadii

    pts1 = pts1 + blkRadii +1
    pts1 = pts1.astype(int)
    pts2 = pts2 + blkRadii +1
    pts2 = pts2.astype(int)

    blkSize = 2 * blkRadii +1
    npts = pts1.shape[1]
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


    v_new = np.array([])

    for j in np.arange(Igradient_fd):

        featureimg = featureimg_total[:,:,j]
        dfeatureimg = featureimg.astype('double')
        v = np.array([])
        itimg = np.cumsum(featureimg,axis=0)
        itimg = np.cumsum(itimg,axis=1)
        iimg = np.zeros((itimg.shape[0]+2,itimg.shape[1]+2))
        iimg[1:-1,1:-1] = itimg

        # tempa = dfeatureimg**2
        # tempb = np.ones((blkRadii*2+1,blkRadii*2+1))
        # print(tempa.shape,tempb.shape)
        # tempc = np.correlate(tempa,tempb)
        # print(tempc.shape)
        # normMat = np.sqrt(np.correlate(dfeatureimg**2,np.ones((blkRadii*2+1,blkRadii*2+1)),mode='same'))
        normMat = cv2.filter2D(dfeatureimg**2, -1, np.ones((blkRadii*2+1,blkRadii*2+1)),borderType=cv2.BORDER_CONSTANT)
        normMat = np.sqrt(normMat)
        normMat = normMat[blkRadii:-blkRadii,blkRadii:-blkRadii]
        idx_zero = np.where(normMat == 0)
        normMat[idx_zero] = 1e-10
  
        for i in np.arange(npts):

            mbSize = np.floor((i + samp_per_scale) / samp_per_scale)
            mbSize = np.int(mbSize)
    
            iiPt1 = iimg[pts1[0,i]+mbSize-1 : pts1[0,i]+effr+mbSize , pts1[1,i]+mbSize-1 : pts1[1,i]+effc+mbSize]
            iiPt2 = iimg[pts1[0,i]+mbSize-1 : pts1[0,i]+effr+mbSize , pts1[1,i]-1        : pts1[1,i]+effc]
            iiPt3 = iimg[pts1[0,i]-1        : pts1[0,i]+effr        , pts1[1,i]+mbSize-1 : pts1[1,i]+effc+mbSize]
            iiPt4 = iimg[pts1[0,i]-1        : pts1[0,i]+effr        , pts1[1,i]-1        : pts1[1,i]+effc]

            blockSum1 = iiPt4 + iiPt1 - iiPt2 - iiPt3


            iiPt1 = iimg[pts2[0,i]+mbSize-1 : pts2[0,i]+effr+mbSize , pts2[1,i]+mbSize-1 : pts2[1,i]+effc+mbSize]
            iiPt2 = iimg[pts2[0,i]+mbSize-1 : pts2[0,i]+effr+mbSize , pts2[1,i]-1        : pts2[1,i]+effc]
            iiPt3 = iimg[pts2[0,i]-1        : pts2[0,i]+effr        , pts2[1,i]+mbSize-1 : pts2[1,i]+effc+mbSize]
            iiPt4 = iimg[pts2[0,i]-1        : pts2[0,i]+effr        , pts2[1,i]-1        : pts2[1,i]+effc]

            blockSum2 = iiPt4 + iiPt1 - iiPt2 - iiPt3


            blockSum1 = blockSum1/(mbSize*mbSize)
            blockSum2 = blockSum2/(mbSize*mbSize)

            diffImg = (blockSum1 - blockSum2) / normMat

            selectedGrid = diffImg[::gridSpacing,::gridSpacing]
            selectedGrid = selectedGrid.T.reshape(-1,1)
            if v.shape[0] == 0:
                v = selectedGrid
            else:
                v = np.hstack((v,selectedGrid))

        if v_new.shape[0] == 0:
            v_new = v
        else:
            v_new = np.hstack((v_new,v))
            
    return v_new