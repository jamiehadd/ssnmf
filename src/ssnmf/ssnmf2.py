#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Class and functions for training (SS)NMF model.
    
    The NMF model consists of the data matrix to be factorized, X, the factor matrices, A and
    S.  Each model also consists of a label matrix, Y, classification factor matrix, B, and
    classification weight parameter, lam (although these three variables will be empty if Y is not
    input).  It will also include binary missing data/label indicator matrices W and L (these will
    be all ones if not provided).  These parameters define the objective function defining the model: 
    (1) ||X - AS||_F^2 or 
    (2) D(X||AS) or
    (3) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 or 
    (4) ||X - AS||_F^2 + lam * D(Y||BS) or
    (5) D(X||AS) + lam * ||Y - BS||_F^2 or
    (6) D(X||AS) + lam * D(Y||BS).

    Examples
    --------

'''

import numpy as np
from numpy import linalg as la

class SSNMF:
    
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
        self.modelNum = kwargs.get('modelNum',1)        #initialize model indicator
        self.W = kwargs.get('W',np.ones(rows,cols))     #initialize missing data indicator matrix
        self.A = kwargs.get('A',np.random.rand(rows,k)) #initialize factor A
        self.S = kwargs.get('S',np.random.rand(k,cols)) #initialize factor S

        #check dimensions of X, A, and S match
        if rows != np.shape(self.A)[0]:
            raise Exception('The row dimensions of X and A are not equal.')
        if cols != np.shape(self.S)[1]:
            raise Exception('The column dimensions of X and S are not equal.')
        if np.shape(self.A)[1] != k:
            raise Exception('The column dimension of A is not k.')
        if np.shape(self.S)[0] != k:
            raise Exception('The row dimension of S is not k.')
        
        #supervision initializations (optional)
        self.Y = kwargs.get('Y',None)
        if self.Y is not None:
            #check dimensions of X and Y match
            if np.shape(self.Y)[1] != np.shape(self.X)[1]:
                raise Exception('The column dimensions of X and Y are not equal.')
            
            classes = np.shape(self.Y)[0]
            self.B = kwargs.get('B',np.random.rand(classes,k))
            self.lam = kwargs.get('lam',1)
            self.L = kwargs.get('L',np.ones(classes,cols)) #initialize missing label indicator matrix

            #check dimensions of Y, S, and B match
            if np.shape(self.B)[0] != classes:
                raise Exception('The row dimensions of Y and B are not equal.')
            if np.shape(self.B)[1] != k:
                raise Exception('The column dimension of B is not k.')
        else:
            self.B = None
            self.lam = None
            self.L = None
                       
    def mult(self,**kwargs):
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
        
        if saveerrs:
            # based on model number, initialize correct type of error array(s)
    
        for i in range(numiters):
            #multiplicative updates for A, S, and possibly B
            # based on model number, use proper update functions for A,S,(B)
            
        
            if saveerrs:
               # based on model number, initialize correct type of error array(s) 
        
        if saveerrs:
            # based on model number, return correct type of error array(s)

    def dictupdateFro(self):
        #update to use for A and B in Fro norm errors (and S in model (1))

    def dictupdateIdiv(self):
        #update to use for A and B in I-div errors (and S in model (2))

    def repupdateFF(self):
        #update to use for S in model (3)

    def repupdateIF(self):
        #update to use for S in models (4) and (5)

    def repupdateII(self):
        #update to use for S in model (6)
        
    

    def accuracy(self,**kwargs):
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

        Returns
        -------
        acc : float_
            Fraction of correctly classified data points (computed with Y, B, S).
        '''

        Y = kwargs.get('Y',self.Y)
        B = kwargs.get('B',self.B)
        S = kwargs.get('S',self.S)

        if Y is None:
            raise Exception('Label matrix Y not provided: model is not supervised.')
        
        numdata = np.shape(Y)[1]

        #count number of data points which are correctly classified
        numacc = 0
        Yhat = B @ S
        for i in range(numdata):
            true_max = np.argmax(Y[:,i])
            approx_max = np.argmax(Yhat[:,i])

            if true_max == approx_max:
                numacc = numacc + 1

        #return fraction of correctly classified data points
        acc = numacc/numdata
        return acc

    def Idiv(self,Z,D,S,**kwargs):
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

        #compute divergence
        Zhat = D @ S
        div = np.multiply(Z, np.log(np.divide(Z+eps, Zhat+eps))) - Z + Zhat
        Idiv = np.sum(np.sum(div))
        return Idiv

    def fronorm(self,Z,D,S,**kwargs):
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

        Returns
        -------
        fronorm : float_
            Frobenius norm between Z and DS.
        '''

        if Z is None:
            raise Exception('Matrix Z not provided.')

        #compute norm
        Zhat = D @ S
        fronorm = np.linalg.norm(Z - Zhat,'fro')
        return fronorm


