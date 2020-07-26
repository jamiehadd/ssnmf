# Import necessary packages
import numpy as np
import torch
#import torchvision
import matplotlib.pyplot as plt
from time import time
import os
#from google.colab import drive
import scipy.optimize.nnls as nnls
from numpy import linalg as la

import ssnmf


class Evaluation:
    def __init__(self, train_features, train_labels, test_features, test_labels, k, lam, numiters):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.k = k
        self.m = np.shape(train_features)[0]
        self.train_n = np.shape(train_features)[1]
        self.test_n = np.shape(test_features)[1]
        self.c = np.shape(train_labels)[0]
        self.lam = lam
        self.numiters = numiters
        self.train_model = ssnmf.SSNMF(X = self.train_features, k = self.k, Y = self.train_labels, lam = self.lam)

    def eval(self):
        ''' Apply the SSNMF Method '''
        train_model_error = self.train_model.mult(numiters = self.numiters, saveerrs = True)
        
        ## Compute the final classification accuracy for the model (i.e., the accuracy computed at the last iteration)
        train_acc = self.train_model.accuracy()

        #compute test model and evaluate
        S_test = self.test_model(self.train_model.A,self.train_model.B)
        test_acc = self.test_accuracy(self.train_model.B,S_test)
        test_reconerr = la.norm(self.test_features - self.train_model.A @ S_test, 'fro')
        test_classerr = la.norm(self.test_labels - self.train_model.B @ S_test, 'fro')
        test_err = test_reconerr**2 + self.lam * test_classerr**2

        return train_model_error, train_acc, self.numiters, [test_err, test_reconerr, test_classerr, test_acc], S_test

    def test_model(self,A_train,B_train):
        S_test = np.zeros([self.k,self.test_n])
        for i in range(self.test_n):
            S_test[:,i] = nnls(A_train,self.test_features[:,i])[0]

        return S_test


    """
    To-do: replace test_kldiv and test_accuracy with SSNMF functions, add model number to init function and add to eval function
    """