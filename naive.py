
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

# 2. Convert text into numerical data
cv = CountVectorizer(stop_words='english')
X_train = cv.fit_transform(train.data)
X_test = cv.transform(test.data)

y_train = train.target
y_test = test.target

# 3. Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred1 = mnb.predict(X_test)

# 4. Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train > 0, y_train)
pred2 = bnb.predict(X_test > 0)

# 5. Evaluate both models
print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1, average='weighted'))

print("\nBernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1 Score:", f1_score(y_test, pred2, average='weighted'))
