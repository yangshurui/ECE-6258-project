import scipy.io as sio
import numpy as np

from sklearn.svm import LinearSVC

ttt = 0

path = ['mat/1','mat/2a','mat/2b']
n = [10,11,11]
c = [81,432,432]

imdb = sio.loadmat(path[ttt]+'/imdb.mat')

# meta = imdb['meta'][0,0]
# classes = meta['classes'][0]
# sets = meta['sets'][0]
# print(classes)
# print(sets)

images = imdb['images']
# iid = images['id']
# iname = images['name']
# iset = images['set'][0,0][0]
y = images['class'][0,0][0]
# print(iclass.shape)

descrs = sio.loadmat(path[ttt]+'/descrs.mat')
descrs = descrs['descrs'].T
print(descrs.shape)

# y = np.zeros(n[ttt])
# for i in range(1,c[ttt]):
# 	yy = i*np.ones(n[ttt])
# 	y = np.hstack((y,yy))


train = sio.loadmat(path[ttt]+'/train.mat')
train = train['train'][0].T - 1
# print(train.shape)

test = sio.loadmat(path[ttt]+'/test.mat')
test = test['test'][0].T - 1
# print(test.shape)

x_train = descrs[train]
y_train = y[train]
x_test = descrs[test]
y_test = y[test]

# print(x_train.shape)
# print(y_train.shape)

print('ok')

s = 0
ss = 0
for i in range(c[ttt]):
	print(i)
	y_train1 = np.where(y_train == i,1,-1)
	y_test1 = np.where(y_test == i,1,-1)
	if ((y_train1 == -np.ones(y_train1.shape)).all()):
		continue
	clf=LinearSVC(C=1e9)
	clf.fit(x_train,y_train1)
	y_predict = clf.predict(x_test)
	ans = np.where(y_predict == y_test1,1,0)
	# print(ans)
	s = s + np.sum(ans) / ans.shape[0]
	ss = ss + 1
	print(np.sum(ans))
print(s/ss*100)

print('ok')

clf=LinearSVC(C=1e9)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
ans = np.where(y_predict == y_test,1,0)
print(np.sum(ans)/ans.shape[0]*100)
