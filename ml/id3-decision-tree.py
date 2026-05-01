from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

d = load_iris()
X = d.data
y = d.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

m = DecisionTreeClassifier(criterion='entropy')
m.fit(X_train, y_train)

y_pred = m.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1:", f1_score(y_test, y_pred, average='macro'))

plt.figure(figsize=(10,6))
plot_tree(m)
plt.show()