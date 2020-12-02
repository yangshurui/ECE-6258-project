import numpy as np
import computeCoordinates 

datasetList = np.array(['kth-tips'])
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
crossValIndex = 1

meanAcc = np.zeros((datasetList.shape[0],1))
pmAcc = np.zeros((datasetList.shape[0],1))

for dataIndex in np.arange(datasetList.shape[0]):
    resAcc = np.zeros((1,crossValIndex))
    for validationIndex in np.arange(crossValIndex):
        opts = {}
        opts['prefix'] = ('DmdVectorVlad-%s-seed-%d-scale-%d') % (datasetList[dataIndex], validationIndex, dmdOpts['scale'])
        opts['experimentDir'] = 'experiments'
        opts['dataset'] = datasetList[dataIndex]
        opts['datasetDir'] = 'data\\' + datasetList[dataIndex] ;
        opts['resultDir'] = opts['experimentDir'] + '\\' + opts['prefix']
        opts['imdbPath'] = opts['resultDir'] + '\\imdb.mat'
        opts['encoderPath'] = opts['resultDir'] + '\\encoder.mat'
        opts['diaryPath'] = opts['resultDir'] + '\\diary.txt'
        opts['cacheDir'] = opts['resultDir'] + '\\cache'
        opts['seed'] = validationIndex
        opts['C'] = 10

        # res_mAcc = testDmdVlad(opts, dmdOpts, fvOpts)

        # resAcc[validationIndex] = res['mAcc'] * 100;

    # if (crossValIndex.shape != 1):
        # resAcc = resAcc[crossValIndex]
    
    # meanAcc[dataIndex] = np.mean(resAcc)
    # pmAcc[dataIndex] = np.std(resAcc, ddof=1)
