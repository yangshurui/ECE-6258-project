import scipy.io as sio
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

ttt = 1

path = ['mat/1','mat/2a','mat/2b']
n = [10,11,11]
c = [81,432,432]
imdb = sio.loadmat(path[ttt]+'/imdb.mat')
images = imdb['images']
y = images['class'][0,0][0]
descrs = sio.loadmat(path[ttt]+'/descrs.mat')
descrs = descrs['descrs'].T
train = sio.loadmat(path[ttt]+'/train.mat')
train = train['train'][0].T - 1
test = sio.loadmat(path[ttt]+'/test.mat')
test = test['test'][0].T - 1
x_train = descrs[train]
y_train = y[train]
x_test = descrs[test]
y_test = y[test]
print('ok')
s = 0
ss = 0
for i in range(c[ttt]):
	y_train1 = np.where(y_train == i,1,-1)
	y_test1 = np.where(y_test == i,1,-1)
	if ((y_train1 == -np.ones(y_train1.shape)).all()):
		continue	
	clf = LinearSVC()
	clf.fit(x_train,y_train1)
	y_predict = clf.predict(x_test)
	ans = np.where(y_predict == y_test1,1,0)
	s = s + np.sum(ans) / ans.shape[0]
	ss = ss + 1
	print(i)
print(s/ss*100)
print('ok')
clf = LinearSVC()
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
ans = np.where(y_predict == y_test,1,0)
print(np.sum(ans)/ans.shape[0]*100)
