#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:49:09 2021

@author: madushani
"""

"""Run movie review experiment on scale data."""


import os
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0,'./experiments/movie_review/')
from prep_regression_dat import *
from methods_regression import Methods
from sklearn.metrics import r2_score
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import max_error
from statistics import mean
from statistics import stdev
#import matplotlib.pyplot as plt

np.random.seed(1)

# ------------ PARAMETERS ------------------------------------------------
rank = 13 # (int) input rank for NMF and (S)SNMF models
iterations = 3 # (odd int) number of iterations to run for analysis
nmf_search = 1 # (boolean) run for various tolerance values
ssnmf_search = 0 # (boolean) run for various tolerances and regularizers 
Run_Linear_regression = 1 # (boolean) run linear regression on the TFIDF representation of documents
# ------------------------------------------------------------------------


## Load scale data
## We create a list of all reviews and a list of all corresponding ratings
## Note that both lists contain strings and therefore we have to convert
## ratings to float for our regression problem

## Path to data
path = 'scale_data/scaledata'
list_dir = os.listdir(path)

reviews_list = list() ## A list of all reviews
for direc in list_dir:
    file = os.path.join(path, direc, "subj." + direc) 
    reviews_file = open(file)
    reviews_file_contents = reviews_file.read()
    reviews_list.extend(reviews_file_contents.splitlines())
    reviews_file.close()


ratings_list = list() ## A list of all ratings
for direc in list_dir:
    file = os.path.join(path, direc, "rating." + direc) 
    ratings_file = open(file)
    ratings_file_contents = ratings_file.read()
    ratings_list.extend(ratings_file_contents.splitlines())
    ratings_file.close()
    
## Convert ratings to float    
ratings_list = np.array(ratings_list)
ratings_list = ratings_list.astype('float64') 
## Transform response to be in real line (Note that original ratings are in [0, 1])
ratings_list = np.log(ratings_list + 10)
ratings_list = list(ratings_list)

## Check if the outcome looks normally distributed
# plt.hist(ratings_list, density=True, bins=30) 
# plt.show()

## Train-Test Split 
full_reviews_list_train, full_reviews_list_test, full_ratings_list_train, \
    ratings_list_test = train_test_split(reviews_list, ratings_list, \
                                         test_size=0.30, random_state=42)

## Train-Validation Split
reviews_list_train, reviews_list_val, ratings_list_train, ratings_list_val\
    = train_test_split(full_reviews_list_train, full_ratings_list_train, \
                       test_size=0.25, random_state=1)     

# Compute the TFIDF representation of the train set
vectorizer_train, feature_names_train, X_train = tfidf_train(reviews_list_train, n_features = 5000)
train_ratings = np.array(ratings_list_train)
X_train, train_ratings = shuffle_data(X_train, train_ratings)

# Apply TFIDF transformation to validation set
X_val = tfidf_transform(vectorizer_train, reviews_list_val)
val_ratings = np.array(ratings_list_val)
X_val, val_ratings = shuffle_data(X_val, val_ratings)

# Compute the TFIDF representation of the full train set
vectorizer_train_full, feature_names_train_full, X_train_full = tfidf_train(full_reviews_list_train, n_features = 5000)
full_train_ratings = np.array(full_ratings_list_train)
X_train_full, full_train_ratings = shuffle_data(X_train_full, full_train_ratings)

# Apply TFIDF transformation to test data set
X_test = tfidf_transform(vectorizer_train_full, full_reviews_list_test)
test_ratings = np.array(ratings_list_test)
X_test, test_ratings = shuffle_data(X_test, test_ratings)

## Convert rating 1-d rating arrays to 2-D to use in SSNMF
train_ratings = train_ratings[np.newaxis, :]
val_ratings = val_ratings[np.newaxis, :]
full_train_ratings = full_train_ratings[np.newaxis, :]
test_ratings = test_ratings[np.newaxis, :]


 # ------------------------------------------------------------------------
if ssnmf_search == 1:
    """ Run SSNMF for various tolerance and regularizer values."""

    tol_list = [1e-4,1e-3,1e-2]
    lam_list = [1e+1,1e+2,1e+3]
    mean_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
    std_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
    
    for lam_idx in range (len(lam_list)):
        ssnmf_lam = lam_list[lam_idx]
        print("Testing lambda equal to {}.".format(ssnmf_lam))
        for tol_idx in range (len(tol_list)):
            ssnmf_tol = tol_list[tol_idx]
            print("Testing tolerance equal to {}.".format(ssnmf_tol))
            acc_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
            # Construct an evaluation module
            evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_val,\
                                y_train = train_ratings, y_val = val_ratings,\
                                y_test = val_ratings, X_train_full = X_train,\
                                y_train_full = train_ratings)
    
            for j in range(iterations):
                print("Iteration {}.".format(j))
                for i in range(3,7):
                    # Run SSNMF
                    test_evals, A, B, ssnmf_predicted, ssnmf_iter, S, S_test = evalualtion_module.SSNMF(modelNum = i,
                                                                            ssnmf_tol = ssnmf_tol,lamb = ssnmf_lam, ka = rank, itas= 50)
                    true_ratings = np.squeeze(val_ratings)
                    ssnmf_predicted = np.squeeze(ssnmf_predicted)
                    ## Calculate R**2
                    acc_dict["Model" + str(i)].append(r2_score(true_ratings, ssnmf_predicted))
                    
                    ## Calculate MSE
                    ## acc_dict["Model" + str(i)].append(mean_squared_error(true_ratings, ssnmf_predicted))
                    
                    ## Calculate MAE
                    ## acc_dict["Model" + str(i)].append(mean_absolute_error(true_ratings, ssnmf_predicted))
        
            for i in range(3,7):
                acc = acc_dict["Model" + str(i)]
                mean_dict["Model" + str(i)].append(mean(acc))
                std_dict["Model" + str(i)].append(stdev(acc))
                print("Model {} average accuracy (with tol = {} and lam = {}): {:.4f} ± {:.4f}.".format(i,ssnmf_tol,ssnmf_lam,mean(acc),stdev(acc)))
    
    for i in range(3,7):
        idx_final = 0
        for lam_idx in range(len(lam_list)):
            ssnmf_lam = lam_list[lam_idx]
            for tol_idx in range (len(tol_list)):
                ssnmf_tol = tol_list[tol_idx]
                m_final = mean_dict["Model" + str(i)][idx_final]
                s_final = std_dict["Model" + str(i)][idx_final]
                print("Model {} average accuracy (with tol = {} and lam = {}): {:.4f} ± {:.4f}.".format(i,ssnmf_tol,ssnmf_lam,m_final,s_final))
                idx_final += 1
        print()

# ------------------------------------------------------------------------

if Run_Linear_regression == 1:
    """ Run linear regression on the TFIDF representation of documents."""
    evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_test,\
                            y_train = train_ratings, y_val = val_ratings,\
                            y_test = test_ratings, X_train_full = X_train_full,\
                            y_train_full = full_train_ratings)
    y_test_pred = evalualtion_module.Linear_regression()
    test_true = np.squeeze(test_ratings)
    r2_val = r2_score(test_true, y_test_pred)
    print("\nR-sq value for linear regression is {}.\n".format(r2_val))

# ------------------------------------------------------------------------
if nmf_search == 1:
    """ Run NMF for various tolerance values."""
    tol_list = [1e-5,1e-4]

    for nmf_model in ["NMF", "I_NMF"]:
        mean_acc = []
        std_acc = []

        for tol_idx in range(len(tol_list)):
            nmf_acc = []
            nmf_tol = tol_list[tol_idx]
            print("Testing tolerance equal to {}.".format(nmf_tol))
            # Construct an evaluation module
            evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_val,\
                            y_train = train_ratings, y_val = val_ratings,\
                            y_test = val_ratings, X_train_full = X_train,\
                            y_train_full = train_ratings)
            

            if nmf_model == "NMF":
                for j in range(iterations):
                    print("Iteration {}.".format(j))
                    W, nmf_LR_predicted, nmf_iter, H, H_test = evalualtion_module.NMF(rank=rank, nmf_tol=nmf_tol, beta_loss = "frobenius")
                    true_ratings = np.squeeze(val_ratings)
                    ## Calculate R**2
                    nmf_acc.append(r2_score(true_ratings, nmf_LR_predicted))
                
                    ## Calculate MSE
                    ## nmf_acc.append(mean_squared_error(true_ratings, nmf_LR_predicted))
                    
                    ## Calculate MAE
                    ## nmf_acc.append(mean_absolute_error(true_ratings, nmf_LR_predicted))
                    

            if nmf_model == "I_NMF":
                for j in range(iterations):
                    print("Iteration {}.".format(j))
                    W, nmf_LR_predicted, nmf_iter, H, H_test = evalualtion_module.NMF(rank=rank, nmf_tol=nmf_tol, beta_loss = "kullback-leibler")
                    true_ratings = np.squeeze(val_ratings)
                    ## Calculate R**2
                    nmf_acc.append(r2_score(true_ratings, nmf_LR_predicted))
                
                    ## Calculate MSE
                    ## nmf_acc.append(mean_squared_error(true_ratings, nmf_LR_predicted))
                    
                    ## Calculate MAE
                    ## nmf_acc.append(mean_absolute_error(true_ratings, nmf_LR_predicted))

            mean_acc.append(mean(nmf_acc))
            std_acc.append(stdev(nmf_acc))

        print("\n\nResults for {} iterations.\n".format(iterations))
        for tol_idx in range(len(tol_list)):
            print(nmf_model + " average accuracy (with tol = {}): {:.4f} ± {:.4f}.".format(tol_list[tol_idx],mean_acc[tol_idx],std_acc[tol_idx]))

# ------------------------------------------------------------------------



