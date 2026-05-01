from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

d = load_iris()
X = d.data
y = d.target

p = PCA(n_components=2)
X_p = p.fit_transform(X)

plt.scatter(X_p[:,0], X_p[:,1], c=y)
plt.title("PCA")
plt.show()