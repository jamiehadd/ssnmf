#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ssnmf.ssnmf_numpy import *
from ssnmf.ssnmf_pytorch import *
import numpy as np
import torch


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

