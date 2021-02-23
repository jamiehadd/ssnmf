#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:25:12 2021

@author: madushani
"""

#import ssnmf
from ssnmf.evaluation  import Evaluation


class Methods:

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, X_train_full,\
                    y_train_full):
        """
        Class for all methods.
        Parameters:
            X_train (ndarray): tfidf train data matrix, shape (vocabulary size, number of train documents)
            X_val (ndarray): tfidf validation data matrix, shape (vocabulary size, number of val documents)
            X_test (ndarray): tfidf test data matrix, shape (vocabulary size, number of test documents)
            y_train (ndarray): outcome of all documents in train set (1, number of train documents)
            y_val (ndarray):  outcome of all documents in val set (1, number of val documents)
            y_test (ndarray):  outcome of all documents in test set (1, number of test documents)
            X_train_full (ndarray): tfidf full train data matrix, shape (vocabulary size, number of full train documents)
            y_train_full (ndarray):  outcome of all documents in full train set (1, number of full train documents)
        """
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.X_train_full = X_train_full
        self.y_train_full = y_train_full

    def SSNMF(self, modelNum, ssnmf_tol, lamb, ka, itas, print_results=0):
        """
        Run (S)SNMF on the TFIDF representation of documents.
        Args:
            modelNum (int): SSNMF model number
            ssnmf_tol (float): tolerance for termanating SSNMF model
            lamb (float): regularization parameter of SSNMF
            ka (int): input rank for SSNMF
            itas (int): maximum number of multiplicative update iterations
            hyp_search (boolean): 0: run hyperparameter search algorithm, 1:otherwise
            print_results (boolean): 1: print classification report, heatmaps, and keywords, 0:otherwise
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
