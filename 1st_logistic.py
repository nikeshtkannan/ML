
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Logistic Regression using MLE
mle = LogisticRegression(penalty=None, max_iter=5000)
mle.fit(X_train, y_train)
y_pred = mle.predict(X_test)

print("MLE Accuracy:", accuracy_score(y_test, y_pred))

# 5. MAP using L1 regularization
l1 = LogisticRegression(penalty="l1", solver="liblinear",
                        C=1.0, max_iter=5000)
l1.fit(X_train, y_train)
y_pred_l1 = l1.predict(X_test)

print("MAP with L1 Accuracy:",
      accuracy_score(y_test, y_pred_l1))

# 6. MAP using L2 regularization
l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=5000)
l2.fit(X_train, y_train)
y_pred_l2 = l2.predict(X_test)

print("MAP with L2 Accuracy:",
      accuracy_score(y_test, y_pred_l2))

# 7. Compare parameter estimates
print("\nFirst 5 MLE coefficients:", mle.coef_[0][:5])
print("First 5 L1 coefficients:", l1.coef_[0][:5])
print("First 5 L2 coefficients:", l2.coef_[0][:5])
