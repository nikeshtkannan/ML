import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = sns.load_dataset('mpg')
data = data[['displacement', 'mpg']].dropna()

X = data[['displacement']]
y = data['mpg']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare different polynomial degrees
for degree in [1, 2, 3]:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    print("Degree:", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    plt.scatter(X_test, y_test)
    plt.scatter(X_test, y_pred)
    plt.title(f"Polynomial Regression - Degree {degree}")
    plt.xlabel("Displacement")
    plt.ylabel("MPG")
    plt.show()