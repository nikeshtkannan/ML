import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

# Ridge Regression
ridge = Ridge()
ridge_param = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge, ridge_param, cv=5)
ridge_grid.fit(X_train, y_train)
ridge_pred = ridge_grid.predict(X_test)

# Lasso Regression
lasso = Lasso(max_iter=10000)
lasso_param = {'alpha': [0.01, 0.1, 1, 10]}
lasso_grid = GridSearchCV(lasso, lasso_param, cv=5)
lasso_grid.fit(X_train, y_train)
lasso_pred = lasso_grid.predict(X_test)

# Calculate Metrics
linear_mse = mean_squared_error(y_test, linear_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)

linear_r2 = r2_score(y_test, linear_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

# Evaluation
print("Linear Regression")
print("MSE:", linear_mse)
print("R2:", linear_r2)

print("\nRidge Regression")
print("Best Alpha:", ridge_grid.best_params_)
print("MSE:", ridge_mse)
print("R2:", ridge_r2)

print("\nLasso Regression")
print("Best Alpha:", lasso_grid.best_params_)
print("MSE:", lasso_mse)
print("R2:", lasso_r2)

# ------------------ Graph 1 : MSE Comparison ------------------

models = ["Linear", "Ridge", "Lasso"]
mse = [linear_mse, ridge_mse, lasso_mse]

plt.figure(figsize=(6,4))
plt.bar(models, mse)
plt.title("MSE Comparison")
plt.xlabel("Regression Models")
plt.ylabel("Mean Squared Error")
plt.show()

# ------------------ Graph 2 : R² Score Comparison ------------------

r2 = [linear_r2, ridge_r2, lasso_r2]

plt.figure(figsize=(6,4))
plt.bar(models, r2)
plt.title("R² Score Comparison")
plt.xlabel("Regression Models")
plt.ylabel("R² Score")
plt.show()