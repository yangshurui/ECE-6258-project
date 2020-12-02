import computeCoordinates 
import computeIGradientDmd 
import dippykit as dip
import numpy as np
import cv2
# import trainEncoder1

blkRadii = 7
nPoints = 20

dmdOpts = {}
xi,yi = computeCoordinates.computeCoordinates(blkRadii, nPoints)
dmdOpts['xi'] = xi
dmdOpts['yi'] = yi
dmdOpts['radii'] = blkRadii
dmdOpts['gridspace'] = 2
dmdOpts['scale'] = 4

fvOpts = {}
fvOpts['numDescrs'] = 500000
fvOpts['numKmeanscluster'] = 128

# trainEncoder1.trainEncoder(fvOpts,dmdOpts)

img = cv2.imread('brussels_downsample_gray_square.jpg')

dmdFeature = computeIGradientDmd.computeIGradientDmd(img,dmdOpts)
np.set_printoptions(suppress=True)
# print(dmdFeature[:5,:5])

print(img.shape)
print(dmdFeature.shape)