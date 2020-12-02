import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = np.load("descrs/vlad_data.npy")
print(X.shape)
Y = np.load("descrs/vlad_label.npy")
scaler = StandardScaler()
x = scaler.fit_transform(X)
print(x.shape)

s = 0

for i in range(10):
	y = np.where(Y == i,1,-1)
	# print(y)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

	clf=LinearSVC(C=1e9)
	clf.fit(x_train,y_train)
	y_predict = clf.predict(x_test)
	ans = np.where(y_predict == y_test,1,0)
	# print(ans)
	s = s + np.sum(ans) / ans.shape[0]
	print(np.sum(ans))
print(s/11)