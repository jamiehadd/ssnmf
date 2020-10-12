#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from ssnmf.ssnmf_numpy import *
#from ssnmf.ssnmf_pytorch import *
import numpy as np
from numpy import linalg as la
import torch


class SSNMF_N:
    """
    Class for (S)NMF model.

    The NMF model consists of the data matrix to be factorized, X, the factor matrices, A and
    S.  Each model also consists of a label matrix, Y, classification factor matrix, B, and
    classification weight parameter, lam (although these three variables will be empty if Y is not
    input).  These parameters define the objective function defining the model:
    (1) ||X - AS||_F^2 or
    (2) D(X||AS) or
    (3) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 or
    (4) ||X - AS||_F^2 + lam * D(Y||BS) or
    (5) D(X||AS) + lam * ||Y - BS||_F^2 or
    (6) D(X||AS) + lam * D(Y||BS).
    ...
    Parameters
    ----------
    X        : array
               Data matrix of size m x n.
    k        : int_
               Number of topics.
    modelNum : int_, optional
               Number indicating which of above models user intends to train (the default is 1).
    A        : array, optional
               Initialization for left factor matrix of X of size m x k (the default is a matrix with
               uniform random entries).
    S        : array, optional
               Initialization for right factor matrix of X of size k x n (the default is a matrix with
               uniform random entries).
    Y        : array, optional
               Label matrix of size p x n (default is None).
    B        : array, optional
               Initialization for left factor matrix of Y of size p x k (the default is a matrix with
               uniform random entries if Y is not None, None otherwise).
    lam      : float_, optional
               Weight parameter for classification term in objective (the default is 1 if Y is not
               None, None otherwise).
    W        : array, optional
               Missing data indicator matrix of same size as X (the defaults is matrix of all ones).
    L        : array, optional
               Missing label indicator matrix of same size as Y (the default is matrix of all ones if
               Y is not None, None otherwise).
    tol      : tolerance for termanating the model

    Methods
    -------
    mult(numiters = 10, saveerrs = True)
        Train the selected model via numiters multiplicative updates.
    accuracy()
        Compute the classification accuracy of supervised model (using Y, B, and S).
    kldiv()
        Compute the KL-divergence, D(Y||BS), of supervised model (using Y, B, and S).
    """

    def __init__(self, X, k, **kwargs):
        self.X = X
        rows = np.shape(X)[0]
        cols = np.shape(X)[1]
        self.modelNum = kwargs.get('modelNum', 1)  # initialize model indicator
        self.W = kwargs.get('W', np.ones((rows, cols)))  # initialize missing data indicator matrix
        self.A = kwargs.get('A', np.random.rand(rows, k))  # initialize factor A
        self.S = kwargs.get('S', np.random.rand(k, cols))  # initialize factor S
        self.tol = kwargs.get('tol', 1e-4) # initialize factor tol

        # check dimensions of X, A, and S match
        if rows != np.shape(self.A)[0]:
            raise Exception('The row dimensions of X and A are not equal.')
        if cols != np.shape(self.S)[1]:
            raise Exception('The column dimensions of X and S are not equal.')
        if np.shape(self.A)[1] != k:
            raise Exception('The column dimension of A is not k.')
        if np.shape(self.S)[0] != k:
            raise Exception('The row dimension of S is not k.')

        # supervision initializations (optional)
        self.Y = kwargs.get('Y', None)
        if self.Y is not None:
            # check dimensions of X and Y match
            if np.shape(self.Y)[1] != np.shape(self.X)[1]:
                raise Exception('The column dimensions of X and Y are not equal.')

            classes = np.shape(self.Y)[0]
            self.B = kwargs.get('B', np.random.rand(classes, k))
            self.lam = kwargs.get('lam', 1)
            self.L = kwargs.get('L', np.ones((classes, cols)))  # initialize missing label indicator matrix

            # check dimensions of Y, S, and B match
            if np.shape(self.B)[0] != classes:
                raise Exception('The row dimensions of Y and B are not equal.')
            if np.shape(self.B)[1] != k:
                raise Exception('The column dimension of B is not k.')
        else:
            self.B = None
            self.lam = None
            self.L = None

    def mult(self, **kwargs):
        '''
        Multiplicative updates for training (SS)NMF model.
        Parameters
        ----------
        numiters : int_, optional
            Number of iterations of updates to run (default is 10).
        saveerrs : bool, optional
            Boolean indicating whether to save model errors during iterations.
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        errs : array, optional
            If saveerrs, returns array of ||X - AS||_F for each iteration (length numiters).
        '''
        numiters = kwargs.get('numiters', 10)
        saveerrs = kwargs.get('saveerrs', False)
        eps = kwargs.get('eps', 1e-10)
        initialErr = 0
        previousErr = 0
        currentErr = 0

        if saveerrs:
        # based on model number, initialize correct type of error array(s)
            if self.modelNum == 1 or self.modelNum == 2:
                errs = []  # initialize error array
            else :
                errs = []  # initialize error array
                reconerrs = []
                classerrs = []
                classaccs = []


        for i in range(numiters):
            # multiplicative updates for A, S, and possibly B
            # based on model number, use proper update functions for A,S,(B)
            if self.modelNum == 1:
                self.A = self.dictupdateFro(self.X, self.A, self.S, self.W, eps)
                self.S = np.transpose(self.dictupdateFro(np.transpose(self.X), np.transpose(self.S), np.transpose(self.A), np.transpose(self.W), eps))

                previousErr = currentErr
                currentErr = self.fronorm(self.X, self.A, self.S, self.W)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 2:
                self.A = self.dictupdateIdiv(self.X, self.A, self.S, self.W, eps)
                self.S = np.transpose(self.dictupdateIdiv(np.transpose(self.X), np.transpose(self.S), np.transpose(self.A), np.transpose(self.W), eps))

                previousErr = currentErr
                currentErr = self.Idiv(self.X, self.A, self.S, self.W)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 3:
                self.A = self.dictupdateFro(self.X, self.A, self.S, self.W, eps)
                self.B = self.dictupdateFro(self.Y, self.B, self.S, self.L, eps)
                self.S = self.repupdateFF(eps)

                previousErr = currentErr
                currentErr = self.fronorm(self.X, self.A, self.S, self.W)**2 + self.lam * (self.fronorm(self.Y, self.B, self.S, self.L)**2)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 4:
                self.A = self.dictupdateFro(self.X, self.A, self.S, self.W, eps)
                self.B = self.dictupdateIdiv(self.Y, self.B, self.S, self.L, eps)
                self.S = self.repupdateIF(eps, self.modelNum)

                previousErr = currentErr
                currentErr = self.fronorm(self.X, self.A, self.S, self.W)**2 + self.lam * self.Idiv(self.Y, self.B, self.S, self.L)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 5:
                self.A = self.dictupdateIdiv(self.X, self.A, self.S, self.W, eps)
                self.B = self.dictupdateFro(self.Y, self.B, self.S, self.L, eps)
                self.S = self.repupdateIF(eps, self.modelNum)

                previousErr = currentErr
                currentErr = self.Idiv(self.X, self.A, self.S, self.W) + self.lam * (self.fronorm(self.Y, self.B, self.S, self.L)**2)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 6:
                self.A = self.dictupdateIdiv(self.X, self.A, self.S, self.W, eps)
                self.B = self.dictupdateIdiv(self.Y, self.B, self.S, self.L, eps)
                self.S = self.repupdateII(eps)

                previousErr = currentErr
                currentErr = self.Idiv(self.X, self.A, self.S, self.W) + self.lam * self.Idiv(self.Y, self.B, self.S, self.L)
                if i == 0:
                    initialErr = currentErr

            if i>0 and (previousErr - currentErr) / initialErr < self.tol:
                break

            if saveerrs:
            # based on model number, initialize correct type of error array(s)
                if self.modelNum == 1:
                    errs.append(self.fronorm(self.X, self.A, self.S, self.W))
                if self.modelNum == 2:
                    errs.append(self.Idiv(self.X, self.A, self.S, self.W))
                if self.modelNum == 3:
                    reconerrs.append(self.fronorm(self.X, self.A, self.S, self.W))
                    classerrs.append(self.fronorm(self.Y, self.B, self.S, self.L))
                    errs.append(reconerrs[i] ** 2 + self.lam * classerrs[i] ** 2)
                    classaccs.append(self.accuracy())
                if self.modelNum == 4:
                    reconerrs.append(self.fronorm(self.X, self.A, self.S, self.W))
                    classerrs.append(self.Idiv(self.Y, self.B, self.S, self.L))
                    errs.append(reconerrs[i] ** 2 + self.lam * classerrs[i])
                    classaccs.append(self.accuracy())
                if self.modelNum == 5:
                    reconerrs.append(self.Idiv(self.X, self.A, self.S, self.W))
                    classerrs.append(self.fronorm(self.Y, self.B, self.S, self.L))
                    errs.append(reconerrs[i] + self.lam * (classerrs[i] ** 2))  # save errors
                    classaccs.append(self.accuracy())
                if self.modelNum == 6:
                    reconerrs.append(self.Idiv(self.X, self.A, self.S, self.W))
                    classerrs.append(self.Idiv(self.Y, self.B, self.S, self.L))
                    errs.append(reconerrs[i] + self.lam * classerrs[i])  # save errors
                    classaccs.append(self.accuracy())

        if saveerrs:
            if self.modelNum == 1 or self.modelNum == 2:
                errs =np.array(errs)
                return [errs]
            else:
                errs = np.array(errs)
                reconerrs = np.array(reconerrs)
                classerrs = np.array(classerrs)
                classaccs = np.array(classaccs)
                return [errs, reconerrs, classerrs, classaccs]

    # based on model number, return correct type of error array(s)

    def dictupdateFro(self, Z, D, R, M, eps):
        '''
        multiplicitive update for D and R in ||Z - DR||_F^2
        Parameters
        ----------
        Z   : array
              Data matrix.
        D   : array
              Left factor matrix of Z.
        R   : array
              Right factor matrix of Z.
        M   : array
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        updated D or the transpose of updated R
        '''

        return  np.multiply(
                np.divide(D, eps + np.multiply(M, D@R) @ np.transpose(R)), \
                np.multiply(M, Z) @ np.transpose(R))

    def dictupdateIdiv(self, Z, D, R, M, eps):
        '''
        multiplicitive update for D and R in D(Z||DR)
        Parameters
        ----------
        Z   : array
              Data matrix.
        D   : array
              Left factor matrix of Z.
        R   : array
              Right factor matrix of Z.
        M   : array
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        updated D or the transpose of updated R
        '''
        return np.multiply(np.divide(D, eps + M @ np.transpose(R)), \
                           np.multiply(np.divide(np.multiply(M, Z), eps + np.multiply(M, D @ R)), M) @ np.transpose(R))

    def repupdateFF(self, eps):
    # update to use for S in model (3)
        return np.multiply(
            np.divide(self.S, eps + np.transpose(self.A) @ np.multiply(self.W, self.A @ self.S) + \
                      self.lam * np.transpose(self.B) @ np.multiply(self.L, self.B @ self.S)),
            np.transpose(self.A) @ np.multiply(self.W, self.X) + self.lam * np.transpose(self.B) @ np.multiply(self.L, self.Y)
        )

    def repupdateIF(self, eps, modelNum):
    # update to use for S in models (4) and (5)
        if modelNum == 4:
            return np.multiply(
                np.divide(self.S, eps + (2 * np.transpose(self.A) @ np.multiply(self.W, self.A @ self.S) +
                                         self.lam * np.transpose(self.B) @ self.L)),
                2 * np.transpose(self.A) @ np.multiply(self.W, self.X) + self.lam * np.transpose(self.B) @ \
                np.multiply(np.divide(np.multiply(self.L, self.Y), eps + np.multiply(self.L, self.B @ self.S)), self.L)
            )

        if modelNum == 5:
            return np.multiply(
                np.divide(self.S, eps + np.transpose(self.A) @ self.W + \
                                  2 * self.lam * np.transpose(self.B) @ np.multiply(self.L, self.B @ self.S)),
                np.transpose(self.A) @  np.multiply(np.divide(np.multiply(self.W, self.X), eps + np.multiply(self.W, self.A @ self.S)), self.W) + \
                2 * self.lam * np.transpose(self.B) @ np.multiply(self.L, self.Y)
            )

    def repupdateII(self, eps):
    # update to use for S in model (6)
        return np.multiply(
            np.divide(self.S, eps + np.transpose(self.A) @ self.W + self.lam * np.transpose(self.B) @ self.L),
            np.transpose(self.A) @ np.multiply(np.divide(np.multiply(self.W, self.X), eps + np.multiply(self.W, self.A @ self.S)), self.W) + \
            self.lam * np.transpose(self.B) @ np.multiply(np.divide(np.multiply(self.L, self.Y), eps + np.multiply(self.L, self.B @ self.S)), self.L)
        )

    def accuracy(self, **kwargs):
        '''
        Compute accuracy of supervised model.
        Parameters
        ----------
        Y : array, optional
            Label matrix (default is self.Y).
        B : array, optional
            Left factor matrix of Y (default is self.B).
        S : array, optional
            Right factor matrix of Y (default is self.S).
        L :
        Returns
        -------
        acc : float_
            Fraction of correctly classified data points (computed with Y, B, S).
        '''

        Y = kwargs.get('Y', self.Y)
        B = kwargs.get('B', self.B)
        S = kwargs.get('S', self.S)

        if Y is None:
            raise Exception('Label matrix Y not provided: model is not supervised.')

        numdata = np.shape(Y)[1]

        # count number of data points which are correctly classified
        numacc = 0
        Yhat = B @ S
        for i in range(numdata):
            true_max = np.argmax(Y[:, i])
            approx_max = np.argmax(Yhat[:, i])

            if true_max == approx_max:
                numacc = numacc + 1

        # return fraction of correctly classified data points
        acc = numacc / numdata
        return acc

    def Idiv(self, Z, D, S, M, **kwargs):
        '''
        Compute I-divergence between Z and DS.
        Parameters
        ----------
        Z   : array
              Data matrix.
        D   : array
              Left factor matrix of Z.
        S   : array
              Right factor matrix of Z.
        M   : array
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        Idiv : float_
            I-divergence between Z and DS.
        '''
        eps = kwargs.get('eps', 1e-10)

        if Z is None:
            raise Exception('Matrix Z not provided.')
        if M is None:
            M = np.ones(Z.shape, dtype = float)

        # compute divergence
        Zhat = np.multiply(M, D @ S)
        div = np.multiply(np.multiply(M, Z), np.log(np.divide(np.multiply(M, Z) + eps, Zhat + eps))) \
              - np.multiply(M, Z) + Zhat
        Idiv = np.sum(np.sum(div))
        return Idiv

    def fronorm(self, Z, D, S, M, **kwargs):
        '''
        Compute Frobenius norm between Z and DS.
        Parameters
        ----------
        Z   : array
              Data matrix.
        D   : array
              Left factor matrix of Z.
        S   : array
              Right factor matrix of Z.
        M   : array
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        Returns
        -------
        fronorm : float_
            Frobenius norm between Z and DS.
        '''

        if Z is None:
            raise Exception('Matrix Z not provided.')
        if M is None:
            M = np.ones(Z.shape, dtype = float)

        # compute norm
        Zhat = np.multiply(M, D @ S)
        fronorm = np.linalg.norm(np.multiply(M, Z) - Zhat, 'fro')
        return fronorm



class SSNMF_T:
    """
    Class for (S)NMF model.

    The NMF model consists of the data matrix to be factorized, X, the factor matrices, A and
    S.  Each model also consists of a label matrix, Y, classification factor matrix, B, and
    classification weight parameter, lam (although these three variables will be empty if Y is not
    input).  These parameters define the objective function defining the model:
    (1) ||X - AS||_F^2 or
    (2) D(X||AS) or
    (3) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 or
    (4) ||X - AS||_F^2 + lam * D(Y||BS) or
    (5) D(X||AS) + lam * ||Y - BS||_F^2 or
    (6) D(X||AS) + lam * D(Y||BS).
    ...
    Parameters
    ----------
    X        : torch.Tensor
               Data matrix of size m x n.
    k        : int_
               Number of topics.
    modelNum : int_, optional
               Number indicating which of above models user intends to train (the default is 1).
    A        : torch.Tensor, optional
               Initialization for left factor matrix of X of size m x k (the default is a matrix with
               uniform random entries).
    S        : torch.Tensor, optional
               Initialization for right factor matrix of X of size k x n (the default is a matrix with
               uniform random entries).
    Y        : torch.Tensor, optional
               Label matrix of size p x n (default is None).
    B        : torch.Tensor, optional
               Initialization for left factor matrix of Y of size p x k (the default is a matrix with
               uniform random entries if Y is not None, None otherwise).
    lam      : float_, optional
               Weight parameter for classification term in objective (the default is 1 if Y is not
               None, None otherwise).
    W        : torch.Tensor, optional
               Missing data indicator matrix of same size as X (the defaults is matrix of all ones).
    L        : torch.Tensor, optional
               Missing label indicator matrix of same size as Y (the default is matrix of all ones if
               Y is not None, None otherwise).
    tol      : tolerance for termanating the model.

    Methods
    -------
    mult(numiters = 10, saveerrs = True)
        Train the selected model via numiters multiplicative updates.
    accuracy()
        Compute the classification accuracy of supervised model (using Y, B, and S).
    kldiv()
        Compute the KL-divergence, D(Y||BS), of supervised model (using Y, B, and S).
    """

    def __init__(self, X, k, **kwargs):
        cuda = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.X = X
        rows = X.size(0)
        cols = X.size(1)
        self.modelNum = kwargs.get('modelNum', 1)  # initialize model indicator
        self.W = kwargs.get('W', torch.ones([rows, cols], dtype=torch.float, device=cuda))  # initialize missing data indicator matrix
        self.A = kwargs.get('A', torch.rand(rows, k, dtype=torch.float, device=cuda))  # initialize factor A
        self.S = kwargs.get('S', torch.rand(k, cols, dtype=torch.float, device=cuda))  # initialize factor S
        self.tol = kwargs.get('tol', 1e-4)  # initialize factor tol

        # check dimensions of X, A, and S match
        if rows != self.A.size(0):
            raise Exception('The row dimensions of X and A are not equal.')
        if cols != self.S.size(1):
            raise Exception('The column dimensions of X and S are not equal.')
        if self.A.size(1) != k:
            raise Exception('The column dimension of A is not k.')
        if self.S.size(0) != k:
            raise Exception('The row dimension of S is not k.')

        # supervision initializations (optional)
        self.Y = kwargs.get('Y', None)
        if self.Y is not None:
            # check dimensions of X and Y match
            if self.Y.size(1) != self.X.size(1):
                raise Exception('The column dimensions of X and Y are not equal.')

            classes = self.Y.size(0)
            self.B = kwargs.get('B', torch.rand(classes, k, dtype=torch.float, device=cuda))
            self.lam = kwargs.get('lam', 1)
            self.L = kwargs.get('L', torch.ones([classes, cols], dtype=torch.float, device=cuda))  # initialize missing label indicator matrix

            # check dimensions of Y, S, and B match
            if self.B.size(0) != classes:
                raise Exception('The row dimensions of Y and B are not equal.')
            if self.B.size(1) != k:
                raise Exception('The column dimension of B is not k.')
        else:
            self.B = None
            self.lam = None
            self.L = None

    def mult(self, **kwargs):
        '''
        Multiplicative updates for training (SS)NMF model.
        Parameters
        ----------
        numiters : int_, optional
            Number of iterations of updates to run (default is 10).
        saveerrs : bool, optional
            Boolean indicating whether to save model errors during iterations.
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        errs : array, optional
            If saveerrs, returns array of ||X - AS||_F for each iteration (length numiters).
        '''
        numiters = kwargs.get('numiters', 10)
        saveerrs = kwargs.get('saveerrs', False)
        eps = kwargs.get('eps', 1e-10)
        initialErr = 0
        previousErr = 0
        currentErr = 0

        if saveerrs:
        # based on model number, initialize correct type of error array(s)
            if self.modelNum == 1 or self.modelNum == 2:
                errs = []  # initialize error array
            else :
                errs = []  # initialize error array
                reconerrs = []
                classerrs = []
                classaccs = []


        for i in range(numiters):
            # multiplicative updates for A, S, and possibly B
            # based on model number, use proper update functions for A,S,(B)
            if self.modelNum == 1:
                self.A = self.dictupdateFro(self.X, self.A, self.S, self.W, eps)
                self.S = torch.t(self.dictupdateFro(torch.t(self.X), torch.t(self.S), torch.t(self.A), torch.t(self.W), eps))

                previousErr = currentErr
                currentErr = self.fronorm(self.X, self.A, self.S, self.W)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 2:
                self.A = self.dictupdateIdiv(self.X, self.A, self.S, self.W, eps)
                self.S = torch.t(self.dictupdateIdiv(torch.t(self.X), torch.t(self.S), torch.t(self.A), torch.t(self.W), eps))

                previousErr = currentErr
                currentErr = self.Idiv(self.X, self.A, self.S, self.W)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 3:
                self.A = self.dictupdateFro(self.X, self.A, self.S, self.W, eps)
                self.B = self.dictupdateFro(self.Y, self.B, self.S, self.L, eps)
                self.S = self.repupdateFF(eps)

                previousErr = currentErr
                currentErr = self.fronorm(self.X, self.A, self.S, self.W)**2 + self.lam * (self.fronorm(self.Y, self.B, self.S, self.L)**2)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 4:
                self.A = self.dictupdateFro(self.X, self.A, self.S, self.W, eps)
                self.B = self.dictupdateIdiv(self.Y, self.B, self.S, self.L, eps)
                self.S = self.repupdateIF(eps, self.modelNum)

                previousErr = currentErr
                currentErr = self.fronorm(self.X, self.A, self.S, self.W)**2 + self.lam * self.Idiv(self.Y, self.B, self.S, self.L)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 5:
                self.A = self.dictupdateIdiv(self.X, self.A, self.S, self.W, eps)
                self.B = self.dictupdateFro(self.Y, self.B, self.S, self.L, eps)
                self.S = self.repupdateIF(eps, self.modelNum)

                previousErr = currentErr
                currentErr = self.Idiv(self.X, self.A, self.S, self.W) + self.lam * (self.fronorm(self.Y, self.B, self.S, self.L)**2)
                if i == 0:
                    initialErr = currentErr

            if self.modelNum == 6:
                self.A = self.dictupdateIdiv(self.X, self.A, self.S, self.W, eps)
                self.B = self.dictupdateIdiv(self.Y, self.B, self.S, self.L, eps)
                self.S = self.repupdateII(eps)

                previousErr = currentErr
                currentErr = self.Idiv(self.X, self.A, self.S, self.W) + self.lam * self.Idiv(self.Y, self.B, self.S, self.L)
                if i == 0:
                    initialErr = currentErr

            if i > 0 and (previousErr - currentErr) / initialErr < self.tol:
                break

            if saveerrs:
            # based on model number, initialize correct type of error array(s)
                if self.modelNum == 1:
                    errs.append(self.fronorm(self.X, self.A, self.S, self.W))
                if self.modelNum == 2:
                    errs.append(self.Idiv(self.X, self.A, self.S, self.W))
                if self.modelNum == 3:
                    reconerrs.append(self.fronorm(self.X, self.A, self.S, self.W))
                    classerrs.append(self.fronorm(self.Y, self.B, self.S, self.L))
                    errs.append(reconerrs[i] ** 2 + self.lam * classerrs[i] ** 2)
                    classaccs.append(self.accuracy())
                if self.modelNum == 4:
                    reconerrs.append(self.fronorm(self.X, self.A, self.S, self.W))
                    classerrs.append(self.Idiv(self.Y, self.B, self.S, self.L))
                    errs.append(reconerrs[i] ** 2 + self.lam * classerrs[i])
                    classaccs.append(self.accuracy())
                if self.modelNum == 5:
                    reconerrs.append(self.Idiv(self.X, self.A, self.S, self.W))
                    classerrs.append(self.fronorm(self.Y, self.B, self.S, self.L))
                    errs.append(reconerrs[i] + self.lam * (classerrs[i] ** 2))  # save errors
                    classaccs.append(self.accuracy())
                if self.modelNum == 6:
                    reconerrs.append(self.Idiv(self.X, self.A, self.S, self.W))
                    classerrs.append(self.Idiv(self.Y, self.B, self.S, self.L))
                    errs.append(reconerrs[i] + self.lam * classerrs[i])  # save errors
                    classaccs.append(self.accuracy())

        if saveerrs:
            if self.modelNum == 1 or self.modelNum == 2:
                if torch.cuda.is_available():
                    errs = torch.cuda.FloatTensor(errs)
                else:
                    errs = torch.FloatTensor(errs)
                return [errs]
            else:
                if torch.cuda.is_available():
                    errs = torch.cuda.FloatTensor(errs)
                    reconerrs = torch.cuda.FloatTensor(reconerrs)
                    classerrs = torch.cuda.FloatTensor(classerrs)
                    classaccs = torch.cuda.FloatTensor(classaccs)
                else:
                    errs = torch.FloatTensor(errs)
                    reconerrs = torch.FloatTensor(reconerrs)
                    classerrs = torch.FloatTensor(classerrs)
                    classaccs = torch.FloatTensor(classaccs)
                return [errs, reconerrs, classerrs, classaccs]

    # based on model number, return correct type of error array(s)

    def dictupdateFro(self, Z, D, R, M, eps):
        '''
        multiplicitive update for D and R in ||Z - DR||_F^2
        Parameters
        ----------
        Z   : torch.Tensor
              Data matrix.
        D   : torch.Tensor
              Left factor matrix of Z.
        R   : torch.Tensor
              Right factor matrix of Z.
        M   : torch.Tensor
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        updated D or the transpose of updated R
        '''

        return torch.mul(
               torch.div(D, eps + torch.mul(M, D@R) @ torch.t(R)),
               torch.mul(M, Z) @ torch.t(R))

    def dictupdateIdiv(self, Z, D, R, M, eps):
        '''
        multiplicitive update for D and R in D(Z||DR)
        Parameters
        ----------
        Z   : torch.Tensor
              Data matrix.
        D   : torch.Tensor
              Left factor matrix of Z.
        R   : torch.Tensor
              Right factor matrix of Z.
        M   : torch.Tensor
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        updated D or the transpose of updated R
        '''
        return torch.mul(torch.div(D, eps + M @ torch.t(R)),
                         torch.mul(torch.div(torch.mul(M, Z), eps + torch.mul(M, D @ R)), M) @ torch.t(R))

    def repupdateFF(self, eps):
    # update to use for S in model (3)
        return torch.mul(
            torch.div(self.S, eps + torch.t(self.A) @ torch.mul(self.W, self.A @ self.S) +
                      self.lam * torch.t(self.B) @ torch.mul(self.L, self.B @ self.S)),
            torch.t(self.A) @ torch.mul(self.W, self.X) + self.lam * torch.t(self.B)
            @ torch.mul(self.L, self.Y))

    def repupdateIF(self, eps, modelNum):
    # update to use for S in models (4) and (5)
        if modelNum == 4:
            return torch.mul(
                torch.div(self.S, eps + (2 * torch.t(self.A) @ torch.mul(self.W, self.A @ self.S) +
                                         self.lam * torch.t(self.B) @ self.L)
                          ),
                2 * torch.t(self.A) @ torch.mul(self.W, self.X) + self.lam * torch.t(self.B) @ \
                torch.mul(torch.div(torch.mul(self.L, self.Y), eps + torch.mul(self.L, self.B @ self.S)), self.L)
            )

        if modelNum == 5:
            return torch.mul(
                torch.div(self.S, eps + torch.t(self.A) @ self.W + 2 * self.lam * torch.t(self.B) @ torch.mul(self.L, self.B @ self.S)
                          ),
                torch.t(self.A) @ torch.mul(torch.div(torch.mul(self.W, self.X), eps + torch.mul(self.W, self.A @ self.S)), self.W) + \
                2 * self.lam * torch.t(self.B) @ torch.mul(self.L, self.Y)
            )

    def repupdateII(self, eps):
    # update to use for S in model (6)
        return torch.mul(
            torch.div(self.S, eps + torch.t(self.A) @ self.W + self.lam * torch.t(self.B) @ self.L),
            torch.t(self.A) @ torch.mul(torch.div(torch.mul(self.W, self.X), eps + torch.mul(self.W, self.A @ self.S)), self.W) + \
            self.lam * torch.t(self.B) @ torch.mul(torch.div(torch.mul(self.L, self.Y), eps + torch.mul(self.L, self.B @ self.S)), self.L)
        )

    def accuracy(self, **kwargs):
        '''
        Compute accuracy of supervised model.
        Parameters
        ----------
        Y : torch.tensor, optional
            Label matrix (default is self.Y).
        B : torch.tensor, optional
            Left factor matrix of Y (default is self.B).
        S : torch.tensor, optional
            Right factor matrix of Y (default is self.S).
        Returns
        -------
        acc : float_
            Fraction of correctly classified data points (computed with Y, B, S).
        '''

        Y = kwargs.get('Y', self.Y)
        B = kwargs.get('B', self.B)
        S = kwargs.get('S', self.S)

        if Y is None:
            raise Exception('Label matrix Y not provided: model is not supervised.')

        numdata = Y.size(1)

        # count number of data points which are correctly classified
        numacc = 0
        Yhat = B @ S
        for i in range(numdata):
            true_max = torch.argmax(Y[:, i])
            approx_max = torch.argmax(Yhat[:, i])

            if true_max == approx_max:
                numacc = numacc + 1

        # return fraction of correctly classified data points
        acc = numacc / numdata
        return acc

    def Idiv(self, Z, D, S, M, **kwargs):
        '''
        Compute I-divergence between Z and DS.
        Parameters
        ----------
        Z   : torch.tensor
              Data matrix.
        D   : torch.tensor
              Left factor matrix of Z.
        S   : torch.tensor
              Right factor matrix of Z.
        M   : torch.tensor
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        Idiv : float_
            I-divergence between Z and DS.
        '''
        eps = kwargs.get('eps', 1e-10)

        if Z is None:
            raise Exception('Matrix Z not provided.')
        if M is None:
            M = torch.ones(Z.size(0),Z.size(1), dtype=torch.float, device = Z.device)

        # compute divergence
        Zhat = torch.mul(M, D @ S)
        div = torch.mul(torch.mul(M, Z), torch.log(torch.div(torch.mul(M, Z) + eps, Zhat + eps))) \
              - torch.mul(M, Z) + Zhat
        Idiv = torch.sum(torch.sum(div))
        return Idiv

    def fronorm(self, Z, D, S, M, **kwargs):
        '''
        Compute Frobenius norm between Z and DS.
        Parameters
        ----------
        Z   : torch.tensor
              Data matrix.
        D   : torch.tensor
              Left factor matrix of Z.
        S   : torch.tensor
              Right factor matrix of Z.
        M   : torch.tensor
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        Returns
        -------
        fronorm : float_
            Frobenius norm between Z and DS.
        '''

        if Z is None:
            raise Exception('Matrix Z not provided.')
        if M is None:
            M = torch.ones(Z.size(0), Z.size(1), dtype=torch.float, device=Z.device)

        # compute norm
        Zhat = torch.mul(M, D @ S)
        fronorm = torch.norm(torch.mul(M, Z) - Zhat, p='fro')
        return fronorm



class SSNMF(SSNMF_N, SSNMF_T):
    """
        Class for (S)NMF model.
        This class is inherited from class SSNMF_N and class SSNMF_T. It can take both Numpy array and
        PyTorch tensor when initializing the model.

        The NMF model consists of the data matrix to be factorized, X, the factor matrices, A and
        S.  Each model also consists of a label matrix, Y, classification factor matrix, B, and
        classification weight parameter, lam (although these three variables will be empty if Y is not
        input).  These parameters define the objective function defining the model:
        (1) ||X - AS||_F^2 or
        (2) D(X||AS) or
        (3) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 or
        (4) ||X - AS||_F^2 + lam * D(Y||BS) or
        (5) D(X||AS) + lam * ||Y - BS||_F^2 or
        (6) D(X||AS) + lam * D(Y||BS).
        ...
        Parameters
        ----------
        X        : torch.Tensor
                   Data matrix of size m x n.
        k        : int_
                   Number of topics.
        modelNum : int_, optional
                   Number indicating which of above models user intends to train (the default is 1).
        A        : torch.Tensor, optional
                   Initialization for left factor matrix of X of size m x k (the default is a matrix with
                   uniform random entries).
        S        : torch.Tensor, optional
                   Initialization for right factor matrix of X of size k x n (the default is a matrix with
                   uniform random entries).
        Y        : torch.Tensor, optional
                   Label matrix of size p x n (default is None).
        B        : torch.Tensor, optional
                   Initialization for left factor matrix of Y of size p x k (the default is a matrix with
                   uniform random entries if Y is not None, None otherwise).
        lam      : float_, optional
                   Weight parameter for classification term in objective (the default is 1 if Y is not
                   None, None otherwise).
        W        : torch.Tensor, optional
                   Missing data indicator matrix of same size as X (the defaults is matrix of all ones).
        L        : torch.Tensor, optional
                   Missing label indicator matrix of same size as Y (the default is matrix of all ones if
                   Y is not None, None otherwise).
        tol      : Tolerance for relative error stopping criterion
                  (i.e., method stops when difference between consecutive relative errors falls below tol) 
        str      : a flag to indicate whether this model is initialized by Numpy array or PyTorch tensor

        Methods
        -------
        mult(numiters = 10, saveerrs = True)
            Train the selected model via numiters multiplicative updates.
        accuracy()
            Compute the classification accuracy of supervised model (using Y, B, and S).
        kldiv()
            Compute the KL-divergence, D(Y||BS), of supervised model (using Y, B, and S).
        """

    def __init__(self, X, k, **kwargs):
        if type(X) == np.ndarray:
            SSNMF_N.__init__(self, X, k, **kwargs)
            self.str = "numpy"  # protected attribute
        if type(X) == torch.Tensor:
            SSNMF_T.__init__(self, X, k, **kwargs)
            self.str = "torch"  # protected attribute

    def mult(self, **kwargs):
        '''
        Multiplicative updates for training (SS)NMF model.
        Parameters
        ----------
        numiters : int_, optional
            Number of iterations of updates to run (default is 10).
        saveerrs : bool, optional
            Boolean indicating whether to save model errors during iterations.
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        errs : Numpy array or PyTorch tensor, optional
            If saveerrs, returns array of ||X - AS||_F for each iteration (length numiters).
        '''

        if self.str == "numpy":
            return SSNMF_N.mult(self, **kwargs)
        if self.str == "torch":
            return SSNMF_T.mult(self, **kwargs)

    def accuracy(self, **kwargs):
        '''
        Compute accuracy of supervised model.
        Parameters
        ----------
        Y : Numpy array or PyTorch tensor, optional
            Label matrix (default is self.Y).
        B : Numpy array or PyTorch tensor, optional
            Left factor matrix of Y (default is self.B).
        S : Numpy array or PyTorch tensor, optional
            Right factor matrix of Y (default is self.S).
        L :
        Returns
        -------
        acc : float_
            Fraction of correctly classified data points (computed with Y, B, S).
        '''


        if self.str == "numpy":
            return SSNMF_N.accuracy(self, **kwargs)
        if self.str == "torch":
            return SSNMF_T.accuracy(self, **kwargs)

    def Idiv(self, Z, D, S, M, **kwargs):
        '''
        Compute I-divergence between Z and DS.
        Parameters
        ----------
        Z   : Numpy array or PyTorch tensor
              Data matrix.
        D   : Numpy array or PyTorch tensor
              Left factor matrix of Z.
        S   : Numpy array or PyTorch tensor
              Right factor matrix of Z.
        M   : Numpy array or PyTorch tensor
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).
        Returns
        -------
        Idiv : float_
            I-divergence between Z and DS.
        '''
        if self.str == "numpy":
            return SSNMF_N.Idiv(self, Z, D, S, M, **kwargs)
        if self.str == "torch":
            return SSNMF_T.Idiv(self, Z, D, S, M, **kwargs)

    def fronorm(self, Z, D, S, M, **kwargs):
        '''
        Compute Frobenius norm between Z and DS.
        Parameters
        ----------
        Z   : Numpy array or PyTorch tensor
              Data matrix.
        D   : Numpy array or PyTorch tensor
              Left factor matrix of Z.
        S   : Numpy array or PyTorch tensor
              Right factor matrix of Z.
        M   : Numpy array or PyTorch tensor
              Missing data indicator matrix of same size as Z (the defaults is matrix of all ones).
        Returns
        -------
        fronorm : float_
            Frobenius norm between Z and DS.
        '''
        if self.str == "numpy":
            return SSNMF_N.fronorm(self, Z, D, S, M, **kwargs)
        if self.str == "torch":
            return SSNMF_T.fronorm(self, Z, D, S, M, **kwargs)




