#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:15:00 2021

@author: madushani
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download('stopwords')
#from nltk.corpus import stopwords


def tfidf_train(newsgroups_train, n_features):
    """
    Train a TFIDF vectorizer and compute the TFIDF representation of the train data.

    Args:
        newsgroups_train (ndarray): corpus of all documents from all categories in train set
        n_features (int): vocabulary size
    Returns:
        vectorizer_train (object): trained tfidf vectorizer
        feature_names_train (list): list of features extracted from the trained tfidf vectorizer
        X_train (ndarray): tfidf word-document matrix of train data

    """
    # Extract Tfidf weights
    stop_words_list = nltk.corpus.stopwords.words('english')
    vectorizer_train = TfidfVectorizer(max_features=n_features,
                                    min_df=5, max_df=0.70,
                                    token_pattern = '[a-zA-Z]+',
                                    stop_words = stop_words_list)
    vectors_train = vectorizer_train.fit_transform(newsgroups_train)
    feature_names_train = vectorizer_train.get_feature_names() #features list
    dense_train = vectors_train.todense()

    denselist_train = np.array(dense_train).transpose() # tfidf matrix
    X_train = denselist_train.copy() # train data (tfidf)

    return vectorizer_train, feature_names_train, X_train


def tfidf_transform(vectorizer_train, newsgroups_test):
    """
    Apply TFIDF transformation to test data.

    Args:
        vectorizer_train (object): trained tfidf vectorizer
        newsgroups_test (ndarray): corpus of all documents from all categories in test set
    Returns:
        X_test (ndarray): tfidf word-document matrix of test data
    """

    vectors_test = vectorizer_train.transform(newsgroups_test)
    dense_test = vectors_test.todense()
    denselist_test = np.array(dense_test).transpose()
    X_test = denselist_test.copy()

    return X_test


def shuffle_data(X,y):
    """
    Shuffle data X, labels y

    Args/Returns:
        X (ndarray): data matrix, shape (vocabulary, documents)
        y (ndarray): labels, shape (documents,)
    """
    data = np.row_stack((X, y))
    np.random.shuffle(data.T)
    X = data[:-1,:]
    y = data[-1,:]

    return X, y




