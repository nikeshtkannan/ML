
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# 1. Load dataset
data = fetch_20newsgroups(subset='train')

# 2. Preprocess dataset
cv = CountVectorizer(max_features=1000, stop_words='english')
X = cv.fit_transform(data.data)

# 3. Calculate word counts
counts = np.asarray(X.sum(axis=0)).flatten()
N = counts.sum()
V = len(counts)

# 4. MLE estimation
mle = counts / N

# 5. MAP estimation with different priors
alphas = [1, 2, 10]

print("Comparison of MLE and MAP")

words = cv.get_feature_names_out()

# Display top 10 words
top_indices = np.argsort(mle)[-10:][::-1]

for i in top_indices:
    print("\nWord:", words[i])
    print("MLE:", mle[i])

    for alpha in alphas:
        map_est = (counts[i] + alpha - 1) / (N + V * (alpha - 1))
        print("MAP alpha =", alpha, ":", map_est)
