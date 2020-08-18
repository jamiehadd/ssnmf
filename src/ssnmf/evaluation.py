""" Class for evaluating (S)SNMF models. """


import numpy as np
from numpy import linalg as la
import scipy.optimize.nnls as nnls

import ssnmf2


class Evaluation:
    """
    Class for evaluating (S)SNMF models on train and test data. It contains functions
    to compute the reconstruction errors and classifications accuracies for each model
    on the train and test data.

    Parameters
    ----------
    train_features : array
                     Train data matrix, shape (#features, #train samples)
    test_features  : array
                     Test data matrix, shape (#features, #test samples)
    train_labels   : array
                     Train label matrix, shape (#classes, #train samples)
    test_labels    : array
                     Test label matrix, shape (#classes, #test samples)
    k              : int_
                     Number of topics.
    modelNum       : int_
                     Number indicating which of the (S)SNMF models to train (see ref. below).
    A              : array, optional
                     Initialization for the left factor matrix of the data matrix, shape (#features, k)
                     (the default is a matrix with uniform random entries).
    S              : array, optional
                     Initialization for the right factor matrix of the train data/label matrix,
                     shape (k, #train samples), (the default is a matrix with uniform random entries).
    B              : array, optional
                     Initialization for left factor matrix of the train label matrix, shape (#classes, k),
                     (the default is a matrix with uniform random entries if Y is not None, None otherwise).
    lam            : float_, optional
                     Weight parameter for classification term in objective (the default is 1).
    numiters       : int_,optional
                     Number of iterations of updates to run (default is 10).
    W_train        : array, optional
                     Missing data indicator matrix for the training process, shape (#features, #train samples + #test samples)
                     i.e. the matrix is used to indicate missing entries in the train data, and indicate which test data to mask
                     in the training process (the default is matrix of all ones for train data and all zeros for test data).
    W_test         : array, optional
                     Missing data indicator matrix used in the testing process, shape (#features, #test samples),
                     i.e. the matrix is used to indicate missing entries in the test data,
                     (the default is matrix of all ones for test data).
    L              : array, optional
                     Missing label indicator matrix for the training process, shape (#classes, #train samples + #test samples)
                     (the default is matrix of all ones for train data and all zeros for test data).
    tol            : float_
                     tolerance for termanating the model (default is 1e-4).
    iter_s         : int_
                     Number of iterations of updates to run to approximate the representaion of the test
                     data when the I-divervene a discrepancy measure for data reconstruction (default is 10).


    Model Numbers
    ----------
    (3) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 or
    (4) ||X - AS||_F^2 + lam * D(Y||BS) or
    (5) D(X||AS) + lam * ||Y - BS||_F^2 or
    (6) D(X||AS) + lam * D(Y||BS)


    Methods
    ----------
    eval()
        Fits (S)SNMF model to the train data, and evaluates the performance on the train and test data.
    test_model()
        Compute the representation of the test data given a trained model.
    computeErrors(W)
        Compute the errors of the model on test data.
    iterativeLocalSearch()
        Perform an iterative local search in hyperparameter space for a set of (locally) optimal (k,lambda,iterations).

    """

    def __init__(self, train_features, train_labels, test_features, test_labels, modelNum, k, **kwargs):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.modelNum = modelNum
        self.k = k

        if train_features.shape[1] != train_labels.shape[1]:
            raise Exception('The column dimensions of train_features and train_labels are not equal.')
        if test_features.shape[1] != test_labels.shape[1]:
            raise Exception('The column dimensions of test_features and test_labels are not equal.')

        self.rows = np.shape(train_features)[0] #number of features
        self.cols = np.shape(train_labels)[1] + np.shape(test_labels)[1] #number of samples in the train and test set
        self.classes = np.shape(self.train_labels)[0] #number of classes in the data
        self.lam = kwargs.get('lam',1)
        self.numiters = kwargs.get('numiters',10)
        self.tol = kwargs.get('tol', 1e-4)
        self.iter_s = kwargs.get('iter_s', 10)
        self.A = kwargs.get('A',np.random.rand(self.rows,k)) #initialize factor A
        self.S = kwargs.get('S',np.random.rand(k,self.cols)) #initialize factor S
        self.B = kwargs.get('B', np.random.rand(self.classes, k)) #initialize factor B

        if self.rows != np.shape(self.A)[0]:
            raise Exception('The row dimension of A is not equal to the number of features in the data.')
        if self.k != np.shape(self.A)[1]:
            raise Exception('The column dimension of A is not equal to k.')
        if self.k != np.shape(self.S)[0]:
            raise Exception('The row dimension of S is not equal to k.')
        if self.cols != np.shape(self.S)[1]:
            raise Exception('The column dimension of S is not equal to the number of samples in the train and test data.')
        if np.shape(self.B)[0] != self.classes:
            raise Exception('The row dimension B is not equal not equal to the number of classes.')
        if np.shape(self.B)[1] != self.k:
            raise Exception('The column dimension of B is not k.')

        self.W_train = kwargs.get('W_train', np.concatenate((np.ones(self.train_features.shape),np.zeros(self.test_features.shape)),axis=1))
        if np.shape(self.W_train)[0] != self.rows:
            raise Exception('The row dimension W is not equal to the number of features in the data.')
        if np.shape(self.W_train)[1] != self.cols:
            raise Exception('The column dimension of W is not equal to the number of samples in the train and test data.')

        self.W_test = kwargs.get('W_test', np.ones(self.test_features.shape))

        self.L = kwargs.get('L', np.concatenate((np.ones(self.train_labels.shape),np.zeros(self.test_labels.shape)),axis=1))
        if np.shape(self.L)[0] != self.classes:
            raise Exception('The row dimension of L is not equal to the number of classes in the data.')
        if np.shape(self.L)[1] != self.cols:
            raise Exception('The column dimension of L is not equal to the number of samples in the train and test data.')

        self.model = ssnmf2.SSNMF(X = np.concatenate((self.train_features, self.test_features), axis = 1), k = self.k, \
                                        modelNum = self.modelNum, Y = np.concatenate((self.train_labels, self.test_labels), axis = 1), \
                                        lam = self.lam, L = self.L, W = self.W_train)

    def eval(self):
        '''
        This function fits the (S)SNMF model to the train data, and evaluates the performance of the model on the train and test data.

        Returns:
            train_evals (list): [ndarray(total reconstruction error on train data (model objective function) for each iteration),
                                 ndarray(data reconstruction error on train data for each iteration),
                                 ndarray (label reconstruction error on train data for each iteration),
                                 ndarray(classification accuracy on train data for each iteration)]
            test_evals (list): [float(total reconstruction error on test data (model objective function)),
                                float(data reconstruction error on test data),
                                float(label reconstruction error on test data),
                                float(classification accuracy on test data)]
        '''
        # Fit (S)SNMF model to train data
        train_evals = self.model.mult(numiters = self.numiters, saveerrs = True) #save train data errors

        # Apply (S)SNMF model to test data
        self.S_test = self.test_model() #representation matrix of test data

        # Compute reconstruction errors
        test_evals = self.computeErrors()

        return train_evals, test_evals


    def test_model(self):
        '''
        Given a trained (S)SNMF model i.e. learned data dictionary, A, the function applies the (S)SNMF model
        to the test data to produce the representation of the test data.

        Returns:
            S_test (ndarray): representation matrix of the test data, shape(#topics, #test features)
        '''
        if self.modelNum == 3 or self.modelNum == 4: # Frobenius discrepancy measure on label data (use nonnegative least squares)
            S_test = np.zeros([self.k, np.shape(self.test_features)[1]])
            for i in range(np.shape(self.test_features)[1]):
                S_test[:,i] = nnls(self.model.A, self.test_features[:,i])[0]

        if self.modelNum == 5 or self.modelNum == 6: # I-divergence discrepancy measure on label data (use mult. upd. I-div)
            S_init = np.random.rand(self.k, self.test_features.shape[1])
            for i in range(self.iter_s):
                S_test = np.transpose(self.model.dictupdateIdiv(np.transpose(self.test_features), np.transpose(S_init), \
                                       np.transpose(self.model.A), np.ones(self.test_features.T.shape), eps= 1e-10))
        return S_test


    def computeErrors(self):
        """
        Compute errors and classification accuracy of the model on the test data.

        Returns:
            errs (float): total reconstruction error on test data (model objective function)
            reconerrs (float): data reconstruction error on test data
            classerrs (float): label reconstruction error on test data
            classaccs (float): classification accuracy on test data

        """
        L_test = np.ones(self.test_labels.shape, dtype = float) # missing label indicator matrix of same size as test_features (the default is matrix of all ones)

        if self.modelNum == 3:
            reconerrs = self.model.fronorm(self.test_features, self.model.A, self.S_test, M = self.W_test)
            classerrs = self.model.fronorm(self.test_labels, self.model.B, self.S_test, M = L_test)
            errs = reconerrs ** 2 + self.lam * classerrs ** 2
            classaccs = self.model.accuracy(Y=self.test_labels, B=self.model.B, S=self.S_test, L= L_test)
        if self.modelNum == 4:
            reconerrs = self.model.fronorm(self.test_features, self.model.A, self.S_test, M = self.W_test)
            classerrs = self.model.Idiv(self.test_labels, self.model.B, self.S_test, M = L_test)
            errs = reconerrs ** 2 + self.lam * classerrs
            classaccs = self.model.accuracy(Y=self.test_labels, B=self.model.B, S=self.S_test, L= L_test)
        if self.modelNum == 5:
            reconerrs = self.model.Idiv(self.test_features, self.model.A, self.S_test, M = self.W_test)
            classerrs = self.model.fronorm(self.test_labels, self.model.B, self.S_test, M = L_test)
            errs = reconerrs + self.lam * (classerrs ** 2)
            classaccs = self.model.accuracy(Y=self.test_labels, B=self.model.B, S=self.S_test, L = L_test)
        if self.modelNum == 6:
            reconerrs = self.model.Idiv(self.test_features, self.model.A, self.S_test, M = self.W_test)
            classerrs = self.model.Idiv(self.test_labels, self.model.B, self.S_test, M = L_test)
            errs = reconerrs + self.lam * classerrs
            classaccs = self.model.accuracy(Y=self.test_labels, B=self.model.B, S=self.S_test, L= L_test)

        return [errs, reconerrs, classerrs, classaccs]


"""
TODO/Qs:
- Incorporate W_test when solving nnls to find S_test?
- Add hyperparameter search code here (modify based on new ssnmf.py module).
"""
