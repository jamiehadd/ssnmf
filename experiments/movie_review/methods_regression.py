#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:25:12 2021

@author: madushani
"""
import sys
#import ssnmf
sys.path.insert(0,'./src/')
from ssnmf.evaluation  import Evaluation
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import NMF
from scipy.optimize import nnls as nnls
sys.path.insert(0,'./experiments/20news/')
# import utils_20news
from utils_20news import *

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
            test_evals(list): [float(total reconstruction error (model objective function) on test data),
                                    float(data reconstruction error on test data),
                                    float(label reconstruction error on test data),
                                    float(classification accuracy on test data)]
            eval_module.model.A (ndarray): learnt word dictionary matrix for data reconstruction, shape (words, topics)
            eval_module.model.B (ndarray): learnt dictionary matrix for classification, shape (classes, topics)
            ssnmf_predicted (ndarray): predicted labels for data points in test set
            ssnmf_iter (int): actual number of iterations of SSNMF model
            S (ndarray): document representation matrix for train set, shape (topics, train + test documents)
            S_test (ndarray): document representation matrix for test set, shape (topics, test documents)
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

        S = eval_module.model.S
        S_test = eval_module.S_test

        return test_evals[:-1], eval_module.model.A, eval_module.model.B, ssnmf_predicted, ssnmf_iter, S, S_test
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

        print("\nRunning NMF (" + beta_loss + ") Linear Regression")

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
            nmf = NMF(n_components=rank, init= 'random', tol = self.nmf_tol, beta_loss = beta_loss, solver = 'mu', max_iter = 600)
            # Dictionary matrix, shape (vocabulary, topics)
            W = nmf.fit_transform(self.X_train_full)
            # Representation matrix, shape (topics, documents)
            H = nmf.components_
            # Actual number of iterations
            nmf_iter = nmf.n_iter_


        # Train linear regression on train data
        LR_model = LinearRegression(fit_intercept=True)
        y = np.squeeze(self.y_train_full)
        X = H.T
        LR_model.fit(X, y)
        
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

        return W, nmf_LR_predicted, nmf_iter, H, H_test
