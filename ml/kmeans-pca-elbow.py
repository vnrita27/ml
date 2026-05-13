from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

d = load_iris()
X = d.data

p = PCA(n_components=2)
X_p = p.fit_transform(X)

wcss = []

for i in range(1, 11):
    k = KMeans(n_clusters=i)
    k.fit(X_p)
    wcss.append(k.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

k = KMeans(n_clusters=3)

y_k = k.fit_predict(X_p)

plt.scatter(X_p[:,0], X_p[:,1], c=y_k)
plt.title("K-Means Clustering after PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()