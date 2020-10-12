"""Run 20newsgroups data experiment."""

import os
import numpy as np
import random
import pickle
import pandas as pd
import methods_20news
from methods_20news import Methods
from prep_20news import *
from utils_20news import *
from statistics import mean
from statistics import median
from statistics import stdev

random.seed(1)
np.random.seed(1)

# ------------ PARAMETERS ------------
rank = 13 # (int) input rank for NMF and (S)SNMF models
iterations = 11 # (odd int) number of iterations to run for analysis
run_analysis = 1 # (boolean) run all methods to obtain class. acc./keywords/class. reports/heatmaps (reproduce paper results)
nmf_search = 0 # (boolean) run for various tolerance values (reproduce paper results with iterations = 10)
ssnmf_search = 0 # (boolean) run for various tolerances and regularizers (reproduce paper results with iterations = 10)
clust_analysis = 0 # (boolean) run code to compute clustering scores (reproduce paper results)
# -------------------------------------

cls_names =["Computers","Sciences","Politics","Religion","Recreation"]
sub_names = ["graphics", "mac", "windows", "crypt", "electronics", "space", "guns", "mideast", \
             "atheism", "christian", "autos", "baseball", "hockey"]

# Load and subsample 20news data, assign a label to each category, and split data into train, val, and test sets
newsgroups_train, train_labels, newsgroups_test, test_labels, newsgroups_val, val_labels, train_subcat, test_subcat, val_subcat = load_data()

# Construct a full train set that consists of both train and validation set
X_train_full = newsgroups_train+newsgroups_val
train_labels_full =  np.concatenate((train_labels, val_labels), axis=0)
train_subcat_full =  np.concatenate((train_subcat, val_subcat), axis=0)

# Compute the TFIDF representation of the train set
vectorizer_train, feature_names_train, X_train = tfidf_train(newsgroups_train, n_features = 5000)
X_train, train_labels, train_subcat = shuffle_data(X_train, train_labels, train_subcat)

# Apply TFIDF transformation to validation set
X_val = tfidf_transform(vectorizer_train, newsgroups_val)
X_val, val_labels, val_subcat = shuffle_data(X_val, val_labels, val_subcat)

# Compute the TFIDF representation of the full train set
vectorizer_train_full, feature_names_train_full, X_train_full = tfidf_train(X_train_full, n_features = 5000)
X_train_full, train_labels_full, train_subcat_full = shuffle_data(X_train_full, train_labels_full, train_subcat_full)

# Apply TFIDF transformation to test data set
X_test = tfidf_transform(vectorizer_train_full, newsgroups_test)
X_test, test_labels, test_subcat = shuffle_data(X_test, test_labels, test_subcat)

if run_analysis == 1:
    # Construct an evaluation module
    evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_test,\
                            train_labels = train_labels, val_labels = val_labels,\
                            test_labels = test_labels, X_train_full = X_train_full,\
                            train_labels_full = train_labels_full, cls_names = cls_names,\
                            feature_names_train=feature_names_train_full)

    # "Optimal" parameters for SSNMF Models 3,4,5,6 respectively
    ssnmf_tol = [1e-4,1e-4,1e-3,1e-3]
    lamb = [1e+2,1e+1,1e+2,1e+3]
    # "Optimal" NMF parameters
    nmf_tol = 1e-4

    # Run SSNMF Analysis
    acc_dict, A_dict, B_dict, S_dict, S_test_dict, Yhat_dict, median_dict, iter_dict = evalualtion_module.run_analysis(ssnmf_tol= ssnmf_tol, \
                                                                    nmf_tol = nmf_tol, lamb=lamb, ka=rank, itas=50, iterations=iterations)

    evalualtion_module.median_results(acc_dict, A_dict, B_dict, Yhat_dict, median_dict, iter_dict)


if nmf_search == 1:
    """ Run NMF for various tolerance values."""
    mean_acc = []
    std_acc = []
    tol_list = [1e-5,1e-4,1e-3,1e-2]

    for tol_idx in range(len(tol_list)):
        nmf_acc = []
        nmf_tol = tol_list[tol_idx]
        # Construct an evaluation module
        evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_val,\
                                train_labels = train_labels, val_labels = val_labels,\
                                test_labels = val_labels, X_train_full = X_train,\
                                train_labels_full = train_labels, cls_names = cls_names,\
                                feature_names_train=feature_names_train)

        for j in range(iterations):
            print(f"Iteration {j}.")
            nmf_svm_acc, W, nn_svm, nmf_svm_predicted, nmf_iter, H, H_test = evalualtion_module.NMF(rank=rank, nmf_tol=nmf_tol)
            nmf_acc.append(nmf_svm_acc)

        mean_acc.append(mean(nmf_acc))
        std_acc.append(stdev(nmf_acc))

    print(f"\n\nResults for {iterations} iterations.\n")
    for tol_idx in range(len(tol_list)):
        print(f"NMF average accuracy (with tol = {tol_list[tol_idx]}): {mean_acc[tol_idx]:.4f} ± {std_acc[tol_idx]:.4f}.")


if ssnmf_search == 1:
    """ Run SSNMF for various tolerance and regularizer values."""

    tol_list = [1e-4,1e-3,1e-2]
    lam_list = [1e+1,1e+2,1e+3]
    mean_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
    std_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}

    for lam_idx in range (len(lam_list)):
        ssnmf_lam = lam_list[lam_idx]
        for tol_idx in range (len(tol_list)):
            ssnmf_tol = tol_list[tol_idx]
            acc_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
            # Construct an evaluation module
            evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_val,\
                                    train_labels = train_labels, val_labels = val_labels,\
                                    test_labels = val_labels, X_train_full = X_train,\
                                    train_labels_full = train_labels, cls_names = cls_names,\
                                    feature_names_train=feature_names_train)

            for j in range(iterations):
                print(f"Iteration {j}.")
                for i in range(3,7):
                    # Run SSNMF
                    test_evals, A, B, ssnmf_predicted, ssnmf_iter, S, S_test = evalualtion_module.SSNMF(modelNum = i,
                                                                        ssnmf_tol = ssnmf_tol,lamb = ssnmf_lam, ka = rank, itas= 50)
                    acc_dict["Model" + str(i)].append(test_evals[-1])

            for i in range(3,7):
                acc = acc_dict["Model" + str(i)]
                mean_dict["Model" + str(i)].append(mean(acc))
                std_dict["Model" + str(i)].append(stdev(acc))
                print(f"Model {i} average accuracy (with tol = {ssnmf_tol} and lam = {ssnmf_lam}): {mean(acc):.4f} ± {stdev(acc):.4f}.")

    for i in range(3,7):
        idx_final = 0
        for lam_idx in range(len(lam_list)):
            ssnmf_lam = lam_list[lam_idx]
            for tol_idx in range (len(tol_list)):
                ssnmf_tol = tol_list[tol_idx]
                m_final = mean_dict["Model" + str(i)][idx_final]
                s_final = std_dict["Model" + str(i)][idx_final]
                print(f"Model {i} average accuracy (with tol = {ssnmf_tol} and lam = {ssnmf_lam}): {m_final:.4f} ± {s_final:.4f}.")
                idx_final += 1
        print()

if clust_analysis == 1:
    """ Compute hard/soft clustering scores."""
    # Clustering type (Parameter):
    clust = "hard" # clust = "soft"

    # Ground-truth matrix:
    subcat_all = np.concatenate((train_subcat_full, test_subcat), axis = None)
    subcat_onehot = pd.get_dummies(subcat_all).T.to_numpy()

    for i in range(subcat_onehot.shape[1]):
        if np.count_nonzero(subcat_onehot[:,i]) != 1:
            print("Warning: a document belong to more than one subcategory or to no subcategory.")

    # Load representation matrices of each method
    S_dict = pickle.load(open("S_dict.pickle", "rb"))
    S_test_dict = pickle.load(open("S_test_dict.pickle", "rb"))
    median_dict = pickle.load(open("median_dict.pickle", "rb"))

    clustering_analysis(subcat_onehot = subcat_onehot, sub_names = sub_names, S_dict = S_dict, S_test_dict = S_test_dict,\
                        median_dict=median_dict, clust=clust, numb_train = X_train_full.shape[1])
