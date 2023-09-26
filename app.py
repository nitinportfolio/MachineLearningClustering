from sklearn.datasets import make_blobs
# Generate isotropic Gaussian blobs for clustering.
import matplotlib.pyplot as plt
from kmeans import KMeans
centroids = [(-5,-5),(5,5)]
cluster_std = [1,1]
# cluster_std - standard deviation of the clusters
X,y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centroids, n_features = 2, random_state=2)
#print(y)
km = KMeans(n_clusters=2, max_iter=100)
y_means = km.fit_predict(X)


plt.scatter(X[y_means==0,0],X[y_means==0,1], color = "red")
plt.scatter(X[y_means==1,0],X[y_means==1,1], color = "blue")
plt.show()