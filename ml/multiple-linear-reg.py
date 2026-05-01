from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

d = load_iris()
X = d.data
y = d.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

m = LinearRegression()
m.fit(X_train, y_train)

y_pred = m.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
