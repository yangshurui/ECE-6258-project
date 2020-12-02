import numpy as np

centers = np.load("descrs/centers.npy")
print("centers: ", centers.shape)
dist = np.load("tmp/dist.npy")
ind = np.load("tmp/ind.npy")
print(dist.shape)
print(ind.shape)
X = np.load("features/aluminium_foil/15-scale_1_im_1_col.npy")
print(X.shape)

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
print(V.shape)
# print(dist[0])
# print(np.sqrt(np.sum( (feature[0]-centers[ind[0]])**2 )))

