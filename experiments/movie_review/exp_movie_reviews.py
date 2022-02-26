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
from statistics import mean
from utils_20news import *
np.random.seed(1)

# ------------ PARAMETERS ------------------------------------------------
rank = 10 # (int) input rank for NMF and (S)SNMF models
iterations = 10 # (odd int) number of iterations to run for analysis
nmf_search = 0 # (boolean) run for various tolerance values
ssnmf_search = 0 # (boolean) run for various tolerances and regularizers
run_exp = 1
Run_Linear_regression = 0 # (boolean) run linear regression on the TFIDF representation of documents
Run_SSNMF_regression = 0
Run_NMF_regression = 0
# ------------------------------------------------------------------------

## Load scale data
## We create a list of all reviews and a list of all corresponding ratings
## Note that both lists contain strings and therefore we have to convert
## ratings to float for our regression problem

## Path to data
path = './experiments/movie_review/scale_data/scaledata'
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
#ratings_list = np.log(ratings_list + 10)
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
vectorizer_train, feature_names_train, X_train = tfidf_train(reviews_list_train, n_features = 8000)
train_ratings = np.array(ratings_list_train)
X_train, train_ratings = shuffle_data(X_train, train_ratings)

# Apply TFIDF transformation to validation set
X_val = tfidf_transform(vectorizer_train, reviews_list_val)
val_ratings = np.array(ratings_list_val)
X_val, val_ratings = shuffle_data(X_val, val_ratings)

# Compute the TFIDF representation of the full train set
vectorizer_train_full, feature_names_train_full, X_train_full = tfidf_train(full_reviews_list_train, n_features = 8000)
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

#------------------------------------------------------------------------
## Run analysis regression on movie data

if run_exp == 1:
    # Construct an evaluation module
    evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_test,\
                            y_train = train_ratings, y_val = val_ratings,\
                            y_test = test_ratings, X_train_full = X_train_full,\
                            y_train_full = full_train_ratings)

    # "Optimal" parameters for SSNMF Models 3,4,5,6 respectively
    ssnmf_tol = [1e-4,1e-4,1e-4,1e-4]
    lamb = [1e+0,1e+0,1e+1,1e+1]
    # "Optimal" NMF parameters
    fro_nmf_tol = 1e-4
    i_nmf_tol = 1e-4

    r2_dict, iter_dict = evalualtion_module.run_analysis(ssnmf_tol = ssnmf_tol, nmf_tol = fro_nmf_tol, i_nmf_tol = i_nmf_tol, lamb = lamb, ka=rank, itas=50, iterations=iterations)

if Run_Linear_regression == 1:
    # Run linear regression on the TFIDF representation of documents
    evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_test,\
                            y_train = train_ratings, y_val = val_ratings,\
                            y_test = test_ratings, X_train_full = X_train_full,\
                            y_train_full = full_train_ratings)
    lr_pred = evalualtion_module.Linear_regression()
    lr_r2, lr_mse, lr_mae  = regression_metrics(test_ratings, lr_pred)

if Run_SSNMF_regression == 1:
    # Run SSNMF on the TFIDF representation of documents
    # "Optimal" parameters for SSNMF Models 3,4,5,6 respectively
    ssnmf_tol = [1e-4,1e-4,1e-4,1e-4]
    lamb = [1e+0,1e+0,1e+1,1e+1]
    for i in range(3,7):
        evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_test,\
                                y_train = train_ratings, y_val = val_ratings,\
                                y_test = test_ratings, X_train_full = X_train_full,\
                                y_train_full = full_train_ratings)
        ssnmf_predicted, ssnmf_iter = evalualtion_module.SSNMF(modelNum = i, ssnmf_tol = ssnmf_tol[i-3],\
                                                                lamb = lamb[i-3], ka = rank, itas= 50)
        ssnmf_r2, ssnmf_mse, ssnmf_mae  = regression_metrics(test_ratings, ssnmf_predicted)

if Run_NMF_regression == 1:
    # Run SSNMF on the TFIDF representation of documents
    # "Optimal" parameters for SSNMF Models 3,4,5,6 respectively
    fro_nmf_tol = 1e-4
    i_nmf_tol = 1e-4
    evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_test,\
                            y_train = train_ratings, y_val = val_ratings,\
                            y_test = test_ratings, X_train_full = X_train_full,\
                            y_train_full = full_train_ratings)

    #Frobenius
    nmf_LR_predicted, nmf_iter = evalualtion_module.NMF(rank=rank, nmf_tol=fro_nmf_tol, beta_loss = "frobenius")
    nmf_r2, nmf_mse, nmf_mae  = regression_metrics(test_ratings, nmf_LR_predicted)

    # I-divergence
    nmf_LR_predicted, nmf_iter = evalualtion_module.NMF(rank=rank, nmf_tol=i_nmf_tol, beta_loss = "kullback-leibler")
    nmf_r2, nmf_mse, nmf_mae  = regression_metrics(test_ratings, nmf_LR_predicted)

 # ------------------------------------------------------------------------
if ssnmf_search == 1:
    """ Run SSNMF for various tolerance and regularizer values."""

    tol_list = [1e-45,1e-4,1e-3]
    lam_list = [1e+0, 1e+1,1e+2]
    mean_r2_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
    mean_mse_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
    mean_mae_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}

    for lam_idx in range (len(lam_list)):
        ssnmf_lam = lam_list[lam_idx]
        print("Testing lambda equal to {}.".format(ssnmf_lam))
        for tol_idx in range (len(tol_list)):
            ssnmf_tol = tol_list[tol_idx]
            print("Testing tolerance equal to {}.".format(ssnmf_tol))

            r2_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
            mse_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}
            mae_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": []}

            # Construct an evaluation module
            evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_val,\
                                y_train = train_ratings, y_val = val_ratings,\
                                y_test = val_ratings, X_train_full = X_train,\
                                y_train_full = train_ratings)

            for j in range(iterations):
                print("Iteration {}.".format(j))
                for i in range(3,7):
                    # Run SSNMF
                    ssnmf_predicted, ssnmf_iter = evalualtion_module.SSNMF(modelNum = i,
                                                ssnmf_tol = ssnmf_tol,lamb = ssnmf_lam, ka = rank, itas= 50)

                    ssnmf_r2, ssnmf_mse, ssnmf_mae  = regression_metrics(val_ratings, ssnmf_predicted)

                    ## Append Metrics
                    r2_dict["Model" + str(i)].append(ssnmf_r2)
                    mse_dict["Model" + str(i)].append(ssnmf_mse)
                    mae_dict["Model" + str(i)].append(ssnmf_mae)

            for i in range(3,7):
                r2 = r2_dict["Model" + str(i)]
                mean_r2_dict["Model" + str(i)].append(mean(r2))

                mse = mse_dict["Model" + str(i)]
                mean_mse_dict["Model" + str(i)].append(mean(mse))

                mae = mae_dict["Model" + str(i)]
                mean_mae_dict["Model" + str(i)].append(mean(mae))

                print("Model {} averaged metrics (R-squared, MSE, MAE) with tol = {} and lam = {}: ({:.4f}, {:.4f}, {:.4f}).".format(i,ssnmf_tol,ssnmf_lam, mean(r2), mean(mse), mean(mae)))

    for i in range(3,7):
        idx_final = 0
        for lam_idx in range(len(lam_list)):
            ssnmf_lam = lam_list[lam_idx]
            for tol_idx in range (len(tol_list)):
                ssnmf_tol = tol_list[tol_idx]
                r2_final = mean_r2_dict["Model" + str(i)][idx_final]
                mse_final = mean_mse_dict["Model" + str(i)][idx_final]
                mae_final = mean_mae_dict["Model" + str(i)][idx_final]
                print("Model {} averaged metrics (R-squared, MSE, MAE) with tol = {} and lam = {}: ({:.4f}, {:.4f}, {:.4f}).".format(i,ssnmf_tol,ssnmf_lam, r2_final, mse_final, mae_final))
                idx_final += 1
        print()

# ------------------------------------------------------------------------
if nmf_search == 1:
    """ Run NMF for various tolerance values."""
    tol_list = [1e-5,1e-4,1e-3]

    for nmf_model in ["NMF", "I_NMF"]:
        mean_r2 = []
        mean_mse = []
        mean_mae = []

        for tol_idx in range(len(tol_list)):
            nmf_tol = tol_list[tol_idx]
            print("Testing tolerance equal to {}.".format(nmf_tol))
            r2_list  = []
            mse_list = []
            mae_list  = []

            # Construct an evaluation module
            evalualtion_module = Methods(X_train = X_train, X_val = X_val, X_test = X_val,\
                            y_train = train_ratings, y_val = val_ratings,\
                            y_test = val_ratings, X_train_full = X_train,\
                            y_train_full = train_ratings)

            if nmf_model == "NMF":
                for j in range(iterations):
                    print("Iteration {}.".format(j))
                    nmf_LR_predicted, nmf_iter = evalualtion_module.NMF(rank=rank, nmf_tol=nmf_tol, beta_loss = "frobenius")
                    nmf_r2, nmf_mse, nmf_mae  = regression_metrics(val_ratings, nmf_LR_predicted)
                    ## Append Metrics
                    r2_list.append(nmf_r2)
                    mse_list.append(nmf_mse)
                    mae_list.append(nmf_mae)

            if nmf_model == "I_NMF":
                for j in range(iterations):
                    print("Iteration {}.".format(j))
                    nmf_LR_predicted, nmf_iter = evalualtion_module.NMF(rank=rank, nmf_tol=nmf_tol, beta_loss = "kullback-leibler")
                    nmf_r2, nmf_mse, nmf_mae  = regression_metrics(val_ratings, nmf_LR_predicted)
                    ## Append Metrics
                    r2_list.append(nmf_r2)
                    mse_list.append(nmf_mse)
                    mae_list.append(nmf_mae)

            mean_r2.append(mean(r2_list))
            mean_mse.append(mean(mse_list))
            mean_mae.append(mean(mae_list))

        print("\n\nResults for {} iterations.\n".format(iterations))
        for tol_idx in range(len(tol_list)):
            print(nmf_model + " averaged metrics (R-squared, MSE, MAE) with tol = {}: ({:.4f}, {:.4f}, {:.4f}).".format(tol_list[tol_idx],mean_r2[tol_idx],mean_mse[tol_idx],mean_mae[tol_idx]))
