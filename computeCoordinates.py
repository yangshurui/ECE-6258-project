import numpy as np

def computeCoordinates(blkRadii, nPoints):
    blkSize = 2 * blkRadii + 1

    xi = np.array([[],[]])
    yi = np.array([[],[]])

    now_n = 0

    while(now_n < nPoints):
        pts1 = np.round(np.sqrt(blkSize*blkSize/25)*np.random.randn(2,nPoints-now_n))
        pts1[pts1 > blkRadii] = blkRadii
        pts1[pts1 < -blkRadii] = -blkRadii
        xi = np.hstack((xi, pts1))
    
        pts2 = np.round(np.sqrt(blkSize*blkSize/25)*np.random.randn(2,nPoints-now_n))
        pts2[pts2 > blkRadii] = blkRadii
        pts2[pts2 < -blkRadii] = -blkRadii
        yi = np.hstack((yi, pts2))
    
        ids = [i for i in range(xi.shape[1]) if all(xi[:,i] == yi[:,i])]
        xi = np.delete(xi,ids,axis=1)
        yi = np.delete(yi,ids,axis=1)
        now_n = xi.shape[1]

    return xi,yi
