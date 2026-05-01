from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

d = load_iris()
X = d.data
y = d.target

l = LinearDiscriminantAnalysis(n_components=2)
X_l = l.fit_transform(X, y)

plt.scatter(X_l[:,0], X_l[:,1], c=y)
plt.title("LDA")
plt.show()