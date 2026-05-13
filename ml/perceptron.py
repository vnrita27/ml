from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

d = load_iris()

X = d.data[:, :2]
y = d.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

m = Perceptron()

m.fit(X_train, y_train)

y_pred = m.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1:", f1_score(y_test, y_pred, average='macro'))

plt.scatter(X[:,0], X[:,1], c=y)

plt.title("Perceptron")
plt.show()