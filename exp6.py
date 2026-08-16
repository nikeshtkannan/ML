import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Separate input and output
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 1. Logistic Regression WITHOUT Feature Scaling
# -------------------------------------------------

model1 = LogisticRegression(max_iter=1000)

model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

print("WITHOUT FEATURE SCALING")
print("-----------------------")
print("Accuracy :", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1))
print("Recall   :", recall_score(y_test, y_pred1))
print("F1-Score :", f1_score(y_test, y_pred1))


# -------------------------------------------------
# 2. Logistic Regression WITH Feature Scaling
# -------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LogisticRegression(max_iter=1000)

model2.fit(X_train_scaled, y_train)

y_pred2 = model2.predict(X_test_scaled)

print("\nWITH FEATURE SCALING")
print("--------------------")
print("Accuracy :", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2))
print("Recall   :", recall_score(y_test, y_pred2))
print("F1-Score :", f1_score(y_test, y_pred2))


# -------------------------------------------------
# 3. Compare Results
# -------------------------------------------------

print("\nPERFORMANCE COMPARISON")
print("----------------------")
print("                 Accuracy   Precision   Recall   F1-Score")
print("Without Scaling :", 
      round(accuracy_score(y_test, y_pred1), 4),
      round(precision_score(y_test, y_pred1), 4),
      round(recall_score(y_test, y_pred1), 4),
      round(f1_score(y_test, y_pred1), 4))

print("With Scaling    :", 
      round(accuracy_score(y_test, y_pred2), 4),
      round(precision_score(y_test, y_pred2), 4),
      round(recall_score(y_test, y_pred2), 4),
      round(f1_score(y_test, y_pred2), 4))