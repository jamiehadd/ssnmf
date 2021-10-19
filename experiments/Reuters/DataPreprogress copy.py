import pandas as pd
import numpy as np
import nltk
nltk.download('reuters')
nltk.download('stopwords')
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

def load_data(max_f):

    # Extract fileids from the reuters corpus
    fileids = reuters.fileids()

    trn_fileids = list(filter(lambda doc: doc.startswith("train"), fileids))
    tst_fileids = list(filter(lambda doc: doc.startswith("test"), fileids))
    #est_fileids, val_fileids = train_test_split(trn_fileids, test_size=0.25, random_state=42)

    trn_data = get_data(trn_fileids)
    tst_data = get_data(tst_fileids)
    #est_data = get_data(est_fileids)
    #val_data = get_data(val_fileids)

    ### make a dataframe that lables each category an index
    c = reuters.categories()
    l = list(i for i in range(90))

    categories_df = pd.DataFrame({'categories': c, 'labels': l})
    categories_index_df = categories_df.set_index('categories').T

    trn_labels = get_label(trn_data['categories'], categories_index_df)
    tst_labels = get_label(tst_data['categories'], categories_index_df)
    #est_labels = get_label(est_data['categories'], categories_index_df)
    #val_labels = get_label(val_data['categories'], categories_index_df)

    tfidf_vectorizer, X_trn = get_tfidf_vectorizer(trn_data, max_f)
    #X_est = tfidf_transform(tfidf_vectorizer, est_data)
    #X_val = tfidf_transform(tfidf_vectorizer, val_data)
    X_tst = tfidf_transform(tfidf_vectorizer, tst_data)

    #X_est, X_val,
    #est_labels, val_labels
    return X_trn, X_tst, trn_labels, tst_labels


def get_data(fileids):
    # Initialize empty lists to store categories and raw text
    categories = []
    text = []
    # Loop through each file id and collect each files categories and raw text
    for file in fileids:
        categories.append(reuters.categories(file))
        text.append(reuters.raw(file))

    # Combine lists into pandas dataframe. reutersDf is the final dataframe.
    reutersDf = pd.DataFrame({'ids':fileids, 'categories':categories, 'text':text})
    return reutersDf

def get_label(categories_list, df):
    labels = np.zeros((len(reuters.categories()),len(categories_list)))
    for index, mul_label in enumerate(categories_list):
        for label in mul_label:
            labels[df[label], index] = 1
    return labels


def get_tfidf_vectorizer(corpus_data, max_f):
    stop_words_list = nltk.corpus.stopwords.words('english')
    corpus_list = corpus_data["text"].tolist()

    tfidf_vectorizer = TfidfVectorizer(max_features = max_f, token_pattern='[a-zA-Z]+', stop_words=stop_words_list)
    tfidf_vector = tfidf_vectorizer.fit_transform(corpus_list)

    X_train = np.array(tfidf_vector.todense()).transpose()

    return tfidf_vectorizer, X_train


def tfidf_transform(tfidf_vectorizer, corpus_data):
    corpus_list = corpus_data["text"].tolist()

    tfidf_vector = tfidf_vectorizer.fit_transform(corpus_list)

    X_test = np.array(tfidf_vector.todense()).transpose()

    return X_test