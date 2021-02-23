import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

def load_data():
    """
    This function loads 20newsgroups data from sklearn, subsamples the data set, assigns a label to each category, and
    splits data into train, val, and test sets.

    Returns:
        newsgroups_train (list): corpus of all documents from all categories in train set
        train_labels (ndarray): labels of all documents in train set
        newsgroups_test (list): corpus of all documents from all categories in test set
        test_labels (ndarray): labels of all documents in test set
        newsgroups_val (list): corpus of all documents from all categories in val set
        val_labels (ndarray): labels of all documents in val set
        train_subcat (ndarray): subcategory labels of all documents in train set
        test_subcat (ndarray): subcategory labels of all documents in test set
        val_subcat (ndarray): subcategory labels of all documents in val set
    """

    # Load data from categories
    comp = fetch_20newsgroups(subset='all', categories=['comp.graphics', 'comp.sys.mac.hardware', 'comp.windows.x'], \
                shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    science = fetch_20newsgroups(subset='all', categories=['sci.crypt', 'sci.electronics', 'sci.space'], \
                shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    politics = fetch_20newsgroups(subset='all', categories=['talk.politics.guns', 'talk.politics.mideast'], \
                shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    religion = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'soc.religion.christian'], \
                shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    recreation = fetch_20newsgroups(subset='all', categories=['rec.autos', 'rec.sport.baseball', 'rec.sport.hockey'], \
                shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    # Print total number of documents
    data_len = [len(comp.data), len(science.data), len(politics.data), len(recreation.data), len(religion.data)]

    # Subsample classes to create a balanced dataset
    sub_k = min(data_len)
    comp.data, comp.target = [list(t) for t in zip(*random.sample(list(zip(comp.data, comp.target)), sub_k))]
    science.data, science.target  = [list(t) for t in zip(*random.sample(list(zip(science.data, science.target)), sub_k))]
    politics.data, politics.target = [list(t) for t in zip(*random.sample(list(zip(politics.data, politics.target)), sub_k))]
    religion.data, religion.target = [list(t) for t in zip(*random.sample(list(zip(religion.data, religion.target)), sub_k))]
    recreation.data, recreation.target = [list(t) for t in  zip(*random.sample(list(zip(recreation.data, recreation.target)), sub_k))]

    # Subcategories labels
    subcat_comp = np.array(comp.target)
    subcat_scien = np.array(science.target) + len(comp.target_names)
    subcat_polit = np.array(politics.target) + len(comp.target_names) + len(science.target_names)
    subcat_rel = np.array(religion.target) + len(comp.target_names) + len(science.target_names) + len(politics.target_names)
    subcat_rec = np.array(recreation.target) + len(comp.target_names) + len(science.target_names) + len(politics.target_names) + len(religion.target_names)

    # Assign labels to train data based on categories
    y_comp = np.ones(len(comp.data))
    y_scien = 2*np.ones(len(science.data))
    y_polit = 3*np.ones(len(politics.data))
    y_rel = 4*np.ones(len(religion.data))
    y_rec = 5*np.ones(len(recreation.data))
    labels = np.concatenate((y_comp,y_scien,y_polit,y_rel,y_rec), axis=None)

    # Computers
    train_comp, test_comp, y_train_comp, y_test_comp, subcat_comp_train, subcat_comp_test = train_test_split(comp.data, y_comp, subcat_comp, test_size=0.2, random_state=42)
    train_comp, val_comp, y_train_comp, y_val_comp, subcat_comp_train, subcat_comp_val = train_test_split(train_comp, y_train_comp, subcat_comp_train, test_size=0.25, random_state=42)

    # Sciences
    train_scien, test_scien, y_train_scien, y_test_scien, subcat_scien_train, subcat_scien_test = train_test_split(science.data, y_scien, subcat_scien, test_size=0.2, random_state=42)
    train_scien, val_scien, y_train_scien, y_val_scien, subcat_scien_train, subcat_scien_val = train_test_split(train_scien, y_train_scien, subcat_scien_train, test_size=0.25, random_state=42)

    # Politics
    train_polit, test_polit, y_train_polit, y_test_polit, subcat_polit_train, subcat_polit_test = train_test_split(politics.data, y_polit, subcat_polit, test_size=0.2, random_state=42)
    train_polit, val_polit, y_train_polit, y_val_polit, subcat_polit_train, subcat_polit_val = train_test_split(train_polit, y_train_polit, subcat_polit_train, test_size=0.25, random_state=42)

    # Religion
    train_rel, test_rel, y_train_rel, y_test_rel, subcat_rel_train, subcat_rel_test = train_test_split(religion.data, y_rel, subcat_rel, test_size=0.2, random_state=42)
    train_rel, val_rel, y_train_rel, y_val_rel, subcat_rel_train, subcat_rel_val = train_test_split(train_rel, y_train_rel, subcat_rel_train, test_size=0.25, random_state=42)

    # Recreation
    train_rec, test_rec, y_train_rec, y_test_rec, subcat_rec_train, subcat_rec_test = train_test_split(recreation.data, y_rec, subcat_rec,  test_size=0.2, random_state=42)
    train_rec, val_rec, y_train_rec, y_val_rec, subcat_rec_train, subcat_rec_val = train_test_split(train_rec, y_train_rec, subcat_rec_train, test_size=0.25, random_state=42)

    # Corpus from all categories in train set
    newsgroups_train = train_comp + train_scien + train_polit + train_rel + train_rec
    #print(f"Total number of documents in all categories in the train set is {len(newsgroups_train)}.")
    train_labels = np.concatenate((y_train_comp,y_train_scien,y_train_polit,y_train_rel,y_train_rec), axis=None)
    #print(train_labels.shape)
    train_subcat = np.concatenate((subcat_comp_train,subcat_scien_train,subcat_polit_train,subcat_rel_train,subcat_rec_train), axis=None)
    #print(train_subcat.shape)

    # Corpus from all categories in test set
    newsgroups_test = test_comp + test_scien + test_polit + test_rel + test_rec
    test_labels = np.concatenate((y_test_comp,y_test_scien,y_test_polit,y_test_rel,y_test_rec), axis=None)
    test_subcat = np.concatenate((subcat_comp_test,subcat_scien_test,subcat_polit_test,subcat_rel_test,subcat_rec_test), axis=None)

    # Corpus from all categories in validation set
    newsgroups_val = val_comp + val_scien + val_polit + val_rel + val_rec
    val_labels = np.concatenate((y_val_comp,y_val_scien,y_val_polit,y_val_rel,y_val_rec), axis=None)
    val_subcat = np.concatenate((subcat_comp_val,subcat_scien_val,subcat_polit_val,subcat_rel_val,subcat_rec_val), axis=None)

    # Data Split
    total = len(test_labels) + len(val_labels) + len(train_labels)

    return newsgroups_train, train_labels, newsgroups_test, test_labels, newsgroups_val, val_labels, train_subcat, test_subcat, val_subcat


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

def shuffle_data(X,y,z):
    """
    Shuffle data X, labels y, subcategories z.

    Args/Returns:
        X (ndarray): data matrix, shape (vocabulary, documents)
        y (ndarray): labels, shape (documents,)
        z (ndarray): labels, shape (documents,)
    """
    data = np.row_stack((X, y, z))
    np.random.shuffle(data.T)
    X = data[:-2,:]
    y = data[-2,:]
    z = data[-1,:]

    return X, y, z
