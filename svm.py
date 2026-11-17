
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_iris()
X = data.data[:, 2:4]
y = (data.target != 0).astype(int)

# 2. Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Implement Linear SVM
model = SVC(kernel='linear', C=1)
model.fit(X, y)

# 4. Prediction
y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred) * 100)

# 5. Decision Boundary and Margin
x1, x2 = np.meshgrid(
    np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
    np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100)
)

Z = model.decision_function(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)

plt.contour(x1, x2, Z, levels=[-1, 0, 1],
            colors=['red', 'black', 'blue'],
            linestyles=['--', '-', '--'])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='green',
            label='Support Vectors')

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear SVM")
plt.legend()
plt.show()
