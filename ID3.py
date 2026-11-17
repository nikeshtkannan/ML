
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_excel("Online Retail.xlsx")

# 2. Preprocessing
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 3. Group data customer-wise
data = df.groupby('CustomerID').agg(
    TotalQuantity=('Quantity', 'sum'),
    Invoices=('InvoiceNo', 'nunique'),
    TotalSpend=('TotalPrice', 'sum')
)

# 4. Create customer segments
data['Segment'] = pd.qcut(
    data['TotalSpend'], 3,
    labels=['Low', 'Medium', 'High']
)

# 5. Select input and output
X = data[['TotalQuantity', 'Invoices']]
y = data['Segment']

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. ID3 Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X_train, y_train)

# 8. Prediction and accuracy
y_pred = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))

# 9. Feature importance
print("Feature Importance:", model.feature_importances_)

# 10. Visualize decision tree
plot_tree(model, filled=True)
plt.show()
