import string
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC

RANDOM_SEED = 46


def extract_dictionary(df):
    """
    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    for sentence in df['text']:
        # remove punc's
        for p in sentence:
            if p in string.punctuation:
                sentence = sentence.replace(p, ' ')
        # to get lower case
        sentence = sentence.lower()

        # iterate for indexing
        sentence_list = sentence.split()
        for word in sentence_list:
            if word not in word_dict:
                # map to index 0,1,2,...
                word_dict[word] = len(word_dict)

    return word_dict


def generate_feature_matrix(df, word_dict):
    number_of_reviews = df.shape[0]

    data = list(df['text'])
    for i in range(number_of_reviews):
        data[i] = data[i].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    TF_in = TfidfVectorizer(data, stop_words={'english'}, vocabulary=word_dict, sublinear_tf=True)
    feature_matrix = TF_in.fit_transform(data)
    return feature_matrix.toarray()


def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    best_c_val = 0.0
    best_perf = 0.0

    for c in C_range:
        clf = LinearSVC(C=c)
        perf = cross_val_score(clf, X, y, n_jobs=-1)
        perf = perf.mean()
        print("Metric:", metric, ", c:", c, ", cv_perf:", perf)
        if perf > best_perf:
            best_perf = perf
            best_c_val = c
    print("-----------------------------")
    print("Metric:", metric, ", Best c:", best_c_val, ", Best cv_perf:", best_perf)
    return best_c_val, best_perf


def load_data():
    bearish_data = pd.read_csv('training_data/bearish.txt', sep="\n", header=None)
    neutral_data = pd.read_csv('training_data/neutral.txt', sep="\n", header=None)
    bullish_data = pd.read_csv('training_data/bullish.txt', sep="\n", header=None, engine='python')

    bearish_data.columns = ["text"]
    neutral_data.columns = ["text"]
    bullish_data.columns = ["text"]

    bearish_data['sentiment_encode'] = [0] * bearish_data.shape[0]
    neutral_data['sentiment_encode'] = [1] * neutral_data.shape[0]
    bullish_data['sentiment_encode'] = [2] * bullish_data.shape[0]

    print(bearish_data.head(10), bearish_data.shape)
    print(neutral_data.head(10), neutral_data.shape)
    print(bullish_data.head(10), bullish_data.shape)
    df = pd.concat([bearish_data, neutral_data, bullish_data])
    print(df.shape, df.head(10))
    df = df.reset_index(drop=True)
    sns.countplot(df.sentiment_encode)
    plt.xlabel('Sentiment')
    plt.show()
    return df


def get_multiclass_training_data():
    df = load_data()
    df_train, df_test = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=df['sentiment_encode']
    )

    dictionary = extract_dictionary(df_train)
    y_train = df_train['sentiment_encode'].values.copy()
    x_train = generate_feature_matrix(df_train, dictionary)
    y_test = df_test['sentiment_encode'].values.copy()
    x_test = generate_feature_matrix(df_test, dictionary)

    return (x_train, y_train, x_test, y_test, dictionary)


def main():
    np.random.seed(RANDOM_SEED)
    x_train, y_train, x_test, y_test, dictionary = get_multiclass_training_data()
    c_range = [1e-2, 1e-1, 5e-1, 1, 5, 1e1, 1e2]
    best_score = 0
    best_c = 0
    for c in c_range:
        clf = LinearSVC(C=c)
        clf = OneVsRestClassifier(SVC(C=c))
        clf = LinearSVC(penalty='l2', loss='hinge', C=c)
        perf = cross_val_score(clf, x_train, y_train, n_jobs=-1)
        perf = perf.mean()
        if perf > best_score:
            best_c = c
            best_score = perf
        print("c:", c, ", cv_perf:", perf)
    #clf = LinearSVC(C=1)
    #clf = OneVsRestClassifier(SVC(C=10))
    clf = LinearSVC(penalty='l2', loss='hinge', C=best_c)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("F1-Score:", f1_score(y_test, y_pred, average=None))
    print("Accuracy-Score:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
