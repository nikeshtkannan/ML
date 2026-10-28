
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# 1. Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Preprocess the images
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Use a smaller dataset to make the experiment faster
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

# 3. Test different values of K
for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k)

    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(x_test)
    predict_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)

    print("K =", k)
    print("Accuracy:", accuracy * 100, "%")
    print("Training time:", round(train_time, 3), "seconds")
    print("Prediction time:", round(predict_time, 3), "seconds")
    print()
