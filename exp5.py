import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------
# 1. Load 20 Newsgroups Dataset
# ---------------------------------------------------------

print("Loading 20 Newsgroups dataset...")

train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

print("Number of training documents:", len(train_data.data))
print("Number of testing documents :", len(test_data.data))
print("Number of classes            :", len(train_data.target_names))


# ---------------------------------------------------------
# 2. Convert Text into Word Counts
# ---------------------------------------------------------

vectorizer = CountVectorizer(
    stop_words="english",
    min_df=2
)

X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)

y_train = train_data.target
y_test = test_data.target

vocab_size = X_train.shape[1]

print("Vocabulary size:", vocab_size)


# ---------------------------------------------------------
# 3. MLE and MAP Functions
# ---------------------------------------------------------

def estimate_parameters(counts, alpha=0):
    """
    Estimate multinomial parameters.

    alpha = 0  -> MLE
    alpha > 0  -> MAP with symmetric Dirichlet prior
    """

    total_count = counts.sum()

    if alpha == 0:
        # MLE
        probabilities = counts / total_count
    else:
        # MAP
        probabilities = (
            counts + alpha - 1
        ) / (
            total_count + vocab_size * (alpha - 1)
        )

    return probabilities


# ---------------------------------------------------------
# 4. Estimate Parameters for Every Class
# ---------------------------------------------------------

classes = np.unique(y_train)

mle_parameters = {}
map_parameters = {
    0.1: {},
    1.0: {},
    10.0: {}
}

for c in classes:

    # Get documents belonging to this class
    class_documents = X_train[y_train == c]

    # Count words in this class
    word_counts = np.asarray(
        class_documents.sum(axis=0)
    ).flatten()

    # -------------------------
    # MLE
    # -------------------------
    mle_parameters[c] = estimate_parameters(
        word_counts,
        alpha=0
    )

    # -------------------------
    # MAP with different priors
    # -------------------------
    for alpha in map_parameters:

        map_parameters[alpha][c] = estimate_parameters(
            word_counts,
            alpha=alpha
        )


# ---------------------------------------------------------
# 5. Display Parameter Estimates
# ---------------------------------------------------------

words = vectorizer.get_feature_names_out()

print("\nPARAMETER ESTIMATION")
print("--------------------")

# Select first class
c = classes[0]

print("Class:", train_data.target_names[c])

# Display 10 most frequent words
top_words = np.argsort(mle_parameters[c])[-10:][::-1]

print("\nTop words and their probabilities:")

for index in top_words:

    print(
        words[index],
        "\n  MLE      :", mle_parameters[c][index],
        "\n  MAP 0.1  :", map_parameters[0.1][c][index],
        "\n  MAP 1    :", map_parameters[1.0][c][index],
        "\n  MAP 10   :", map_parameters[10.0][c][index]
    )


# ---------------------------------------------------------
# 6. Evaluate Models
# ---------------------------------------------------------

def calculate_log_likelihood(X, y, parameters):

    total_log_likelihood = 0
    total_words = 0

    for i in range(X.shape[0]):

        c = y[i]

        word_counts = X[i].toarray().flatten()

        # Only use words that have positive probability
        positive = parameters[c] > 0

        log_prob = np.zeros_like(parameters[c])

        log_prob[positive] = np.log(
            parameters[c][positive]
        )

        document_log_likelihood = np.sum(
            word_counts[positive] *
            log_prob[positive]
        )

        total_log_likelihood += document_log_likelihood
        total_words += word_counts.sum()

    return total_log_likelihood, total_words


# ---------------------------------------------------------
# 7. MLE Evaluation
# ---------------------------------------------------------

print("\n\nMODEL COMPARISON")
print("----------------")

mle_ll, mle_words = calculate_log_likelihood(
    X_test,
    y_test,
    mle_parameters
)

mle_perplexity = np.exp(
    -mle_ll / mle_words
)

print("\nMLE")
print("Log-Likelihood:", mle_ll)
print("Perplexity    :", mle_perplexity)


# ---------------------------------------------------------
# 8. MAP Evaluation
# ---------------------------------------------------------

for alpha in map_parameters:

    ll, total_words = calculate_log_likelihood(
        X_test,
        y_test,
        map_parameters[alpha]
    )

    perplexity = np.exp(
        -ll / total_words
    )

    print("\nMAP with Dirichlet alpha =", alpha)
    print("Log-Likelihood:", ll)
    print("Perplexity    :", perplexity)


# ---------------------------------------------------------
# 9. Conclusion
# ---------------------------------------------------------

print("\n\nCONCLUSION")
print("----------")

print("""
MLE estimates the multinomial parameters directly from
the observed word frequencies.

MAP estimation introduces a Dirichlet prior. The prior
prevents probabilities from becoming zero and provides
smoothing.

A small alpha gives less smoothing, while a larger alpha
produces stronger smoothing and moves the probabilities
towards a more uniform distribution.

Therefore, changing the Dirichlet prior can significantly
affect the estimated multinomial parameters and the model's
performance.
""")