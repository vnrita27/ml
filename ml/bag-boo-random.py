from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

d = load_iris()
X = d.data[:, :2]
y = d.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

m = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)

m.fit(X_train, y_train)

y_pred = m.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1:", f1_score(y_test, y_pred, average='macro'))

plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Bagging")
plt.show()


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

d = load_iris()
X = d.data[:, :2]
y = d.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

m = AdaBoostClassifier(n_estimators=50)

m.fit(X_train, y_train)

y_pred = m.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1:", f1_score(y_test, y_pred, average='macro'))

plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Boosting")
plt.show()


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

d = load_iris()
X = d.data[:, :2]
y = d.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

m = RandomForestClassifier(n_estimators=100)

m.fit(X_train, y_train)

y_pred = m.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1:", f1_score(y_test, y_pred, average='macro'))

plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Random Forest")
plt.show()