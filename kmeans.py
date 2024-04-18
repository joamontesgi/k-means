from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=10000, centers=3, cluster_std=0.5)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_

silueta = metrics.silhouette_score(X, labels, metric='euclidean')
print(silueta)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', s=200, c='red')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Datos clusterizados usando K-means')
plt.show()