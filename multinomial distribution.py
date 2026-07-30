import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

dataset = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

documents = dataset.data
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents) #fit will learn and transform will convert

word_counts = np.asarray(X.sum(axis=0)).flatten() # np.asarray for sparse matrix,axis=0 for column, flatten()  for 2d to 1d

total_words = np.sum(word_counts)
mle_prob = word_counts / total_words
vocab_size = len(word_counts) #number of unique words

# MLE
mle_prob = word_counts / total_words

# MAP with alpha = 0.1
alpha = 0.1
map_prob_01 = (word_counts + alpha) / (total_words + alpha * vocab_size)

# MAP with alpha = 1
alpha = 1
map_prob_1 = (word_counts + alpha) / (total_words + alpha * vocab_size)

# MAP with alpha = 10
alpha = 10
map_prob_10 = (word_counts + alpha) / (total_words + alpha * vocab_size) # (count+α/total+α​k)

words = vectorizer.get_feature_names_out() # To get the names
results = pd.DataFrame({
    "Word": words,
    "Count": word_counts,
    "MLE": mle_prob,
    "MAP (α=0.1)": map_prob_01,
    "MAP (α=1)": map_prob_1,
    "MAP (α=10)": map_prob_10
})

results = results.sort_values(by="Count", ascending=False)

print(results.head(20))


































# Small α =Trust the data more.
# Large α= Trust the prior more.