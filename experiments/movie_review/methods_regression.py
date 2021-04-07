#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:25:12 2021

@author: madushani
"""
import sys
#import ssnmf
import pickle
sys.path.insert(0,'./src/')
from ssnmf.evaluation  import Evaluation
import numpy as np
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import NMF
from scipy.optimize import nnls as nnls
sys.path.insert(0,'./experiments/20news/')
# import utils_20news
from utils_20news import *
from sklearn.metrics import r2_score, mean_squared_error

class Methods:

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, X_train_full,\
                    y_train_full):
        """
        Class for all methods.
        Parameters:
            X_train (ndarray): tfidf train data matrix, shape (vocabulary size, number of train documents)
            X_val (ndarray): tfidf validation data matrix, shape (vocabulary size, number of val documents)
            X_test (ndarray): tfidf test data matrix, shape (vocabulary size, number of test documents)
            y_train (ndarray): outcome of all documents in train set
            y_val (ndarray):  outcome of all documents in val set
            y_test (ndarray):  outcome of all documents in test set
            X_train_full (ndarray): tfidf full train data matrix, shape (vocabulary size, number of full train documents)
            y_train_full (ndarray):  labels of all documents in full train set
        """
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.X_train_full = X_train_full
        self.y_train_full = y_train_full

    def SSNMF(self, modelNum, ssnmf_tol, lamb, ka, itas):
        """
        Run (S)SNMF on the TFIDF representation of documents.
        Args:
            modelNum (int): SSNMF model number
            ssnmf_tol (float): tolerance for termanating SSNMF model
            lamb (float): regularization parameter of SSNMF
            ka (int): input rank for SSNMF
            itas (int): maximum number of multiplicative update iterations
        Retruns:
            ssnmf_predicted (ndarray): predicted labels for data points in test set
            ssnmf_iter (int): actual number of iterations of SSNMF model
        """

        print("\nRunning SSNMF for Model {}.".format(modelNum))
        self.ssnmf_tol = ssnmf_tol
        self.modelNum = modelNum


        self.opt_ka = ka
        self.opt_lamb = lamb
        self.opt_itas = itas

        eval_module = Evaluation(train_features = self.X_train_full,
                                 train_labels = self.y_train_full,
                                 test_features = self.X_test, test_labels = self.y_test, tol = self.ssnmf_tol,
                                 modelNum = self.modelNum, k = self.opt_ka, lam=self.opt_lamb, numiters = self.opt_itas)

        train_evals, test_evals = eval_module.eval()
        ssnmf_iter = len(train_evals[0])
        ssnmf_predicted = eval_module.model.B@eval_module.S_test

        return ssnmf_predicted, ssnmf_iter

    def Linear_regression(self):

        """
        Run Linear Regression on the TFIDF representation of documents.
        Retruns:
            y_test_pred (ndarray): predicted outcome for data points in test set
        """
        print("\nRunning Linear Regression.")

        LR_model = LinearRegression(fit_intercept=True)
        y = np.squeeze(self.y_train_full)
        X = self.X_train_full.T
        LR_model.fit(X, y)
        y_test_pred = LR_model.predict(self.X_test.T)
        #y_test_pred = np.around(y_test_pred, decimals=1)

        return y_test_pred

    def NMF(self, rank, nmf_tol, beta_loss):
        """
        Run NMF on the TFIDF representation of documents to obtain a low-dimensional representaion (dim=rank),
        then apply linear regression to predict the outcome.
        Args:
            rank (int): input rank for NMF
            nmf_tol (float): tolerance for termanating NMF model
            beta_loss (str): Beta divergence to be minimized (sklearn NMF parameter). Choose 'frobenius', or 'kullback-leibler'.
        Retruns:
             W (ndarray): learnt word dictionary matrix, shape (words, topics)
             nmf_LR_predicted (ndarray): predicted outcomes for data points in test set
             nmf_iter (int): actual number of iterations of NMF
             H (ndarray): document representation matrix for train set, shape (topics, train documents)
             H_test (ndarray): document representation matrix for test set, shape (topics, test documents)
        """
        self.nmf_tol = nmf_tol

        print("\nRunning NMF (" + beta_loss + ") + Linear Regression")

        # TRAINING STEP
        if beta_loss == "frobenius":
            nmf = NMF(n_components=rank, init= 'random', tol = self.nmf_tol, beta_loss = beta_loss, solver = 'mu', max_iter = 400)
            # Dictionary matrix, shape (vocabulary, topics)
            W = nmf.fit_transform(self.X_train_full)
            # Representation matrix, shape (topics, documents)
            H = nmf.components_
            # Actual number of iterations
            nmf_iter = nmf.n_iter_

        if beta_loss == "kullback-leibler":
            nmf = NMF(n_components=rank, init= 'random', tol = self.nmf_tol, beta_loss = beta_loss, solver = 'mu', max_iter = 700)
            # Dictionary matrix, shape (vocabulary, topics)
            W = nmf.fit_transform(self.X_train_full)
            # Representation matrix, shape (topics, documents)
            H = nmf.components_
            # Actual number of iterations
            nmf_iter = nmf.n_iter_

        # Train linear regression on train data
        LR_model = LinearRegression(fit_intercept=True)
        LR_model.fit(H.T, np.squeeze(self.y_train_full))

        # TESTING STEP
        # Compute the representation of test data
        if beta_loss == "frobenius":
            H_test = np.zeros([rank, np.shape(self.X_test)[1]])
            for i in range(np.shape(self.X_test)[1]):
                H_test[:,i] = nnls(W, self.X_test[:,i])[0]

        if beta_loss == "kullback-leibler":
            H_test = np.random.rand(rank, np.shape(self.X_test)[1])
            for i in range(20):
                H_test = np.transpose(dictupdateIdiv(Z = np.transpose(self.X_test), D = np.transpose(H_test), \
                                       R = np.transpose(W), M = np.transpose(np.ones(self.X_test.shape)), eps= 1e-10))

        # Predict test outcomes using trained linear regression model
        nmf_LR_predicted = LR_model.predict(H_test.T)

        return nmf_LR_predicted, nmf_iter

    def run_analysis(self, ssnmf_tol, nmf_tol, i_nmf_tol, lamb, ka, itas, iterations):
        """
        Compute and save all results for each iteration of each model.

        Args:
            ssnmf_tol (list): list of tolerance values for termanating SSNMF Models [3,4,5,6] respecitvely
            nmf_tol (list): tolerance for termanating (frobenius) NMF model
            i_nmf_tol (list): tolerance for termanating (divergence) NMF model
            lamb (list): list of regularization parameter of SSNMF Models [3,4,5,6] respecitvely
            ka (int): input rank for SSNMF
            itas (int): maximum number of multiplicative update iterations
            iterations (int): (odd) number of iterations to run for analysis

        Returns:
            r2_dict (dictionary): for each model, test R-squared value for each iteration (list)
            iter_dict (dictionary): for each model, number of multiplicative updates for each iteration (list) (if applicable)
        """
        print("\nRunning analysis.")

        r2_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": [], "NMF": [], "I_NMF": [], "LR": []}
        iter_dict = {"Model3": [], "Model4": [], "Model5": [], "Model6": [], "NMF": [], "I_NMF": []}

        # Construct an evaluation module
        evalualtion_module = Methods(X_train = self.X_train, X_val = self.X_val, X_test = self.X_test,\
                                y_train = self.y_train, y_val = self.y_val,\
                                y_test = self.y_test, X_train_full = self.X_train_full,\
                                y_train_full = self.y_train_full)

        # Run all methods
        for j in range(iterations):
            print("Iteration {}.".format(j))
            # Run SSNMF
            for i in range(3,7):
                ssnmf_predicted, ssnmf_iter = evalualtion_module.SSNMF(modelNum = i, ssnmf_tol = ssnmf_tol[i-3],lamb = lamb[i-3], ka = ka, itas= itas)
                ssnmf_r2, ssnmf_mse, ssnmf_mae  = regression_metrics(self.y_test, ssnmf_predicted)
                r2_dict["Model" + str(i)].append(ssnmf_r2)
                iter_dict["Model" + str(i)].append(ssnmf_iter)

            # Run LR
            lr_pred = evalualtion_module.Linear_regression()
            lr_r2, lr_mse, lr_mae  = regression_metrics(self.y_test, lr_pred)
            r2_dict["LR"].append(lr_r2)

            # Run NMF + LR
            for nmf_model in ["NMF", "I_NMF"]:
                if nmf_model == "NMF":
                    nmf_LR_predicted, nmf_iter = evalualtion_module.NMF(rank=ka, nmf_tol= nmf_tol, beta_loss = "frobenius")
                if nmf_model == "I_NMF":
                    nmf_LR_predicted, nmf_iter = evalualtion_module.NMF(rank=ka, nmf_tol= i_nmf_tol, beta_loss = "kullback-leibler")

                nmf_r2, nmf_mse, nmf_mae  = regression_metrics(self.y_test, nmf_LR_predicted)
                r2_dict[nmf_model].append(nmf_r2)
                iter_dict[nmf_model].append(nmf_iter)

            # Save all dictionaries
            pickle.dump(r2_dict, open("r2_dict.pickle", "wb"))
            pickle.dump(iter_dict, open("iter_dict.pickle", "wb"))

        # Report average performance for models
        print("\n\nPrinting mean R-squared values ...")
        print("---------------------------------------\n")
        for i in range(3,7):
            r2 = r2_dict["Model" + str(i)]
            print("Model {} average R-square value: {:.4f}.".format(i,mean(r2)))
        r2 = r2_dict["NMF"]
        print("NMF average R-square value: {:.4f}.".format(mean(r2)))
        r2 = r2_dict["I_NMF"]
        print("I_NMF average R-square value: {:.4f}.".format(mean(r2)))
        r2 = r2_dict["LR"]
        print("LR average R-square value: {:.4f}.".format(mean(r2)))

        # Report average number of iterations (mult. updates) for models
        print("\n\nPrinting mean number of iterations (multiplicative updates)...")
        print("----------------------------------------------------------------\n")
        for i in range(3,7):
            iter_list = iter_dict["Model" + str(i)]
            print("Model {} average number of iterations: {:.2f}.".format(i,mean(iter_list)))
        iter_list = iter_dict["NMF"]
        print("NMF average number of iterations: {:.2f}.".format(mean(iter_list)))
        iter_list = iter_dict["I_NMF"]
        print("I_NMF average number of iterations: {:.2f}.".format(mean(iter_list)))

        pickle.dump([ssnmf_tol, nmf_tol, lamb, ka, itas, iterations], open("movie_param_list.pickle", "wb"))

        return r2_dict, iter_dict
