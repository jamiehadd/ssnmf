import pandas as pd

import torch
import nltk
nltk.download('reuters')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import reuters
from nltk.corpus import stopwords

def load_data(file_ids):

    trn_file_ids = list(filter(lambda doc: doc.startswith("train"), file_ids))
    tst_file_ids = list(filter(lambda doc: doc.startswith("test"), file_ids))
    #est_file_ids, val_file_ids = train_test_split(trn_file_ids, test_size=0.25, random_state=42)

    trn_data = get_data(trn_file_ids)
    tst_data = get_data(tst_file_ids)

    multilabel = MultiLabelBinarizer()
    trn_labels = multilabel.fit_transform(trn_data['categories']).T
    tst_labels = multilabel.fit_transform(tst_data['categories']).T

    cuda_device = get_cuda_device()

    trn_labels = torch.from_numpy(trn_labels).to(device=cuda_device)
    tst_labels = torch.from_numpy(tst_labels).to(device=cuda_device)

    return trn_data, tst_data, trn_labels, tst_labels, cuda_device

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

def get_tfidf_matrices(trn_data, tst_data, max_feature, cuda_device):

    tfidf_vectorizer, X_trn = get_tfidf_vectorizer(trn_data, max_feature, cuda_device)
    X_tst = tfidf_transform(tfidf_vectorizer, tst_data, cuda_device)

    return X_trn, X_tst

def get_tfidf_vectorizer(corpus_data, max_feature, cuda_device):
    stop_words_list = nltk.corpus.stopwords.words('english')
    corpus_list = corpus_data["text"].tolist()

    tfidf_vectorizer = TfidfVectorizer(max_features = max_feature, token_pattern='[a-zA-Z]+', stop_words=stop_words_list)
    tfidf_vector = tfidf_vectorizer.fit_transform(corpus_list)

    X_train = torch.t(torch.tensor(tfidf_vector.todense(), dtype=torch.float64, device=cuda_device))

    return tfidf_vectorizer, X_train

def tfidf_transform(tfidf_vectorizer, corpus_data, cuda0):
    """
        Apply TFIDF transformation to test data.
        Args:
            vectorizer_train (object): trained tfidf vectorizer
            newsgroups_test (ndarray): corpus of all documents from all categories in test set
        Returns:
            X_test (ndarray): tfidf word-document matrix of test data
    """
    corpus_list = corpus_data["text"].tolist()

    tfidf_vector = tfidf_vectorizer.fit_transform(corpus_list)

    X_test = torch.t(torch.tensor(tfidf_vector.todense(), dtype=torch.float64, device=cuda0))

    return X_test

def get_cuda_device():
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda:0')
        print('GPU available')
    else:
        cuda_device = torch.device("cpu")
        print('GPU not available. Currently using CPU. In Google Colab Please set GPU via Edit -> Notebook Settings.')

    return cuda_device

def shuffle(X,Y):
    c = torch.randperm(X.shape[1])
    X = X[:, c]
    Y = Y[:, c]
    return X, Y