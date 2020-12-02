import numpy as np
import computeIGradientDmd
import computeCoordinates  
import dippykit as dip
import os
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
import multiprocessing

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

    pool = multiprocessing.Pool(4)

    X_features = pool.map(np.load, X)
    pool.close()
    pool.join()

    return X_features, Y


def trainEncoder(descrs,FVopts):
    clf = MiniBatchKMeans(n_clusters=FVopts['numKmeanscluster'])
    # clf = KMeans(n_clusters=FVopts['numKmeanscluster'])
    clf.fit(descrs)
    centers = clf.cluster_centers_
    labels = clf.labels_

    return centers,labels

class computeVlader(object):
    def __init__(self, tree, centers):
        self.tree = tree
        self.centers = centers
    def __call__(self, x):
        return computeVlad(x, self.tree, self.centers)

def computeVlad(X, tree, centers):
    dist, ind = tree.query(X)
    k = centers.shape[0]
    d = centers.shape[1]
    V = np.zeros([k, d])
    for i in range(k):
        if np.sum(ind[0] == i) > 0:
            li =  np.argwhere(ind[0] == i)
            V[i] = np.sum(X[li,:]-centers[i], axis =0)
    V = V.flatten()
    V = np.sign(V) * np.sqrt(np.abs(V))
    V = V / np.sqrt(np.dot(V, V))

    return V

def computeVlads(X, tree, centers):
    pool = multiprocessing.Pool(4)
    vectors = pool.map(computeVlader(tree, centers), X)
    pool.close()
    pool.join()
    vectors = np.array(vectors)
    return vectors


if __name__=='__main__':
    # arr = np.array(['aluminium_foil','brown_bread','corduroy','cork','cotton',\
    #        'cracker','lettuce_leaf','linen','white_bread','wood','wool'])
    arr = np.array(['aluminium_foil','brown_bread','corduroy','cotton',\
           'cracker','linen','orange_peel','sandpaper','sponge','styrofoam'])
    n = 810

    X,Y = loadFeatures(n,arr)
    print("features load finish")

    fvOpts = {}
    fvOpts['numDescrs'] = 500000
    fvOpts['numKmeanscluster'] = 128

    centers = np.load("descrs/centers.npy")
    print("centers: ", centers.shape)

    tree = KDTree(centers)
    vectors = computeVlads(X, tree, centers)
    np.save("descrs/vlad_data",vectors)
    np.save("descrs/vlad_label",Y)
