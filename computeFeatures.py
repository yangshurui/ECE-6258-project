import os
import numpy as np
import re
import cv2
import computeCoordinates
import computeIGradientDmd

blkRadii = 7
nPoints = 20
dmdOpts = {}
xi,yi = computeCoordinates.computeCoordinates(blkRadii, nPoints)
dmdOpts['xi'] = xi
dmdOpts['yi'] = yi
dmdOpts['radii'] = blkRadii
dmdOpts['gridspace'] = 2
dmdOpts['scale'] = 4

# arr = np.array(['aluminium_foil','brown_bread','corduroy','cork','cotton',\
#        'cracker','lettuce_leaf','linen','white_bread','wood','wool'])
arr = np.array(['aluminium_foil','brown_bread','corduroy','cotton',\
       'cracker','linen','orange_peel','sandpaper','sponge','styrofoam'])

def computeFetures(file):
    for root, dirs, files in os.walk('data/' + file):
        for f in files:
            img = cv2.imread(os.path.join(root, f))
            dmdFeature = computeIGradientDmd.computeIGradientDmd(img, dmdOpts)
            newname = re.findall(r'(.+?)\.png', f)
            newname = 'features/'+ file + '/' + newname[0]
            np.save(newname, dmdFeature)
            # print(newname)

def computeAllFetures():
    for i in range(arr.shape[0]):
        if (not os.path.exists('features/' + arr[i])):
            os.mkdir('features/' + arr[i])
            computeFetures(arr[i])
        print(arr[i])

computeAllFetures()
        