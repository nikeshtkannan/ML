import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load dataset
data = fetch_california_housing()
X = data.data[:, [5]]  # Average number of bedrooms/rooms? Feature: AveRooms is index 3; use AveRooms below
X = data.data[:, [3]]  # Average rooms
y = data.target

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Add a column of ones
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

# 4. Gradient Descent
theta = np.zeros(2)
lr = 0.01
m = len(y_train)

for i in range(1000):
    pred = X_train @ theta
    error = pred - y_train
    theta = theta - lr * (X_train.T @ error) / m

pred1 = X_test @ theta

# 5. Normal Equation
theta2 = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
pred2 = X_test @ theta2

# 6. Evaluate both models
print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, pred1))
print("R2:", r2_score(y_test, pred1))

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, pred2))
print("R2:", r2_score(y_test, pred2))

# 7. Plot the fitted lines
plt.scatter(X_test[:, 1], y_test, s=5, label="Actual data")
plt.plot(X_test[:, 1], pred1, "r.", label="Gradient Descent")
plt.plot(X_test[:, 1], pred2, "g.", label="Normal Equation")
plt.xlabel("Average Rooms")
plt.ylabel("House Price (in $100,000s)")
plt.legend()
plt.show()