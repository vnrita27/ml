from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

d = load_iris()

X = d.data

l = linkage(X, method='ward')

dendrogram(l)

plt.title("Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()