import os
import numpy as np
import re
import dippykit as dip
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

file = "data/wool"

for root, dirs, files in os.walk(file):
    for f in files:
        newname = re.findall(r'(.+?)\.png', f)
        newname = 'features/' + newname[0] + '.npy'
        dmdFeature = np.load(newname)
        print(newname)
        
