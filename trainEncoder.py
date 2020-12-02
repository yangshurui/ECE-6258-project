import numpy as np
import computeIGradientDmd
import computeCoordinates  
import dippykit as dip
import os
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
import multiprocessing


def computeDescrs(x,numDesPerImg,fvOpts):
    tmp = numDesPerImg * np.ones(x.shape[0])
    xx = np.vstack((x,tmp))
    xx = xx.T

    pool = multiprocessing.Pool(4)

    descrs = pool.map(computeDescr, xx)
    pool.close()
    pool.join()

    descrs = np.array(descrs)
    # descrs1 = np.vstack(descrs)

    return descrs

def computeDescr(x):
    dmdFeature = np.load(x[0])
    sel = np.arange(dmdFeature.shape[0])
    np.random.shuffle(sel)
    n = int(float(x[1]))
    sel = sel[0:n]
    descr = dmdFeature[sel,:]

    return descr


def loadFeatures(n,arr):
    X = np.zeros(n)
    X = ["%f" % x for x in X]
    Y = np.zeros(n)
    j = 0
    for i in np.arange(arr.shape[0]):
        for root, dirs, files in os.walk('features/'+arr[i]):
            for f in files:
                mypath = "features/"+arr[i]+'/'+f
                X[j] = mypath
                Y[j] = i
                j = j + 1
    X = np.array(X)
    return X, Y

def trainEncoder(descrs,FVopts):
    clf = MiniBatchKMeans(n_clusters=FVopts['numKmeanscluster'])
    # clf = KMeans(n_clusters=FVopts['numKmeanscluster'])
    clf.fit(descrs)
    centers = clf.cluster_centers_
    labels = clf.labels_

    return centers,labels

if __name__=='__main__':
    # arr = np.array(['aluminium_foil','brown_bread','corduroy','cork','cotton',\
    #        'cracker','lettuce_leaf','linen','white_bread','wood','wool'])
    arr = np.array(['aluminium_foil','brown_bread','corduroy','cotton',\
           'cracker','linen','orange_peel','sandpaper','sponge','styrofoam'])
    n = 810

    fvOpts = {}
    fvOpts['numDescrs'] = 500000
    fvOpts['numKmeanscluster'] = 128

    X,Y = loadFeatures(n,arr)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    print(x_train.shape)
    print(x_test.shape)

    max_descrs = fvOpts['numDescrs']
    numImages = x_train.shape[0]
    numDesPerImg = np.ceil(max_descrs / numImages)
    
    descrs_train = computeDescrs(x_train, numDesPerImg, fvOpts)
    descrs_test = computeDescrs(x_test, numDesPerImg, fvOpts)
    print("descrs compute finish")
    print("train: ", descrs_train.shape)
    print("test: ", descrs_test.shape)
    np.save("descrs/train", descrs_train)
    np.save("descrs/test", descrs_test)
    np.save("descrs/descrs",np.vstack((descrs_train,descrs_test)))

    centers, labels = trainEncoder(np.vstack(descrs_train), fvOpts)
    print("train finish")
    print("centers: ", centers.shape)
    print("labels: ", labels.shape)

    np.save("descrs/centers", centers)
    np.save("descrs/labels", labels)
    
