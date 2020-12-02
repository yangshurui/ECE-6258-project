import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import os

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

arr = np.array(['aluminium_foil','brown_bread','corduroy','cotton',\
       'cracker','linen','orange_peel','sandpaper','sponge','styrofoam'])
# X,Y = loadFeatures(810,arr)
XX = np.load("descrs/descrs.npy")
X = XX[0].reshape(1,-1)
for i in np.arange(1,XX.shape[0]):
    xx = XX[i].reshape(1,-1)
    print(i)
    X = np.vstack((X,xx))
print(X.shape)
Y = np.load("descrs/vlad_label.npy")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

clf=LinearSVC(C=1e9)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print(y_predict[6],y_test[6])
ans = np.where(y_predict == y_test,1,0)
print(ans)