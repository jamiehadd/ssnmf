#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Class and functions for training (S)NMF model.
    
    The NMF model consists of the data matrix to be factorized, X, the factor matrices, A and
    S.  Each model also consists of a label matrix, Y, classification factor matrix, B, and
    classification weight parameter, lam (although these three variables will be empty if Y is not
    input).  These parameters define the objective function defining the model: 
    (1) ||X - AS||_F^2 (train with mult) or 
    (2) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 (train with snmfmult) or 
    (3) ||X - AS||_F^2 + lam * D(Y||BS) (train with klsnmfmult).

    Examples
    --------
    unsupervised (1), saving errors, declaring number of iterations
    
    >>> numIters = 100
    >>> model = SSNMF(numpy.random.rand(100,100),10)
    >>> errs = model.mult(saveerrs = True,numiters = numIters)
    
    unsupervised (1), not saving errors, declaring number of iterations
    
    >>> numIters = 100
    >>> model = SSNMF(numpy.random.rand(100,100),10)
    >>> model.mult(numiters = numIters)
    
    supervised (2), saving errors, default number of iterations
    
    >>> model = SSNMF(data['datamat'], 10, Y = data['labelmat'])
    >>> errs = model.snmfmult(saveerrs = True)
    
    supervised (3), not saving errors, declaring number of iterations

    >>> numIters = 15
    >>> model = SSNMF(data['datamat'], 10, Y = data['labelmat'])
    >>> model.klsnmfmult(numiters = numIters)
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
    (1) ||X - AS||_F^2 (train with mult) or 
    (2) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 (train with snmfmult) or 
    (3) ||X - AS||_F^2 + lam * D(Y||BS) (train with klsnmfmult).

    ...

    Parameters
    ----------
    X : array
        Data matrix of size m x n.
    k : int_
        Number of topics.
    A : array, optional
        Initialization for left factor matrix of X of size m x k (the default is a matrix with
        uniform random entries).
    S : array, optional
        Initialization for right factor matrix of X of size k x n (the default is a matrix with
        uniform random entries).
    Y : array, optional
        Label matrix of size p x n (default is None).
    B : array, optional
        Initialization for left factor matrix of Y of size p x k (the default is a matrix with
        uniform random entries if Y is not None, None otherwise).
    lam : float_, optional
        Weight parameter for classification term in objective (the default is 1 if Y is not
        None, None otherwise).


    Methods
    -------
    mult(numiters = 10, saveerrs = True)
        Train the unsupervised model (1) via numiters multiplicative updates.
    snmfmult(numiters = 10, saveerrs = True)
        Train the supervised model (2) via numiters multiplicative updates.
    klsnmfmult(numiters = 10, saveerrs = True)
        Train the supervised model (3) via numiters multiplicative updates.
    accuracy()
        Compute the classification accuracy of supervised model (using Y, B, and S).
    kldiv()
        Compute the KL-divergence, D(Y||BS), of supervised model (using Y, B, and S).

    """
    def __init__(self, X, k, **kwargs):
        self.X = X
        rows = np.shape(X)[0]
        cols = np.shape(X)[1]
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

            #check dimensions of Y, S, and B match
            if np.shape(self.B)[0] != classes:
                raise Exception('The row dimensions of Y and B are not equal.')
            if np.shape(self.B)[1] != k:
                raise Exception('The column dimension of B is not k.')
        else:
            self.B = None
            self.lam = None
                       
    def mult(self,**kwargs):
        '''
        Multiplicative updates for training unsupervised NMF model (1).

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
            errs = np.empty(numiters) #initialize error array 
    
        for i in range(numiters):
            #multiplicative updates for A and S
            self.A = np.multiply(np.divide(self.A,eps+ self.A @ self.S @ np.transpose(self.S)), \
                                 self.X @ np.transpose(self.S))
            self.S = np.multiply(np.divide(self.S,eps+ np.transpose(self.A) @ self.A @ self.S), \
                                 np.transpose(self.A) @ self.X)
        
            if saveerrs:
                errs[i] = la.norm(self.X - self.A @ self.S, 'fro') #save reconstruction error
        
        if saveerrs:
            return [errs]
        
    def snmfmult(self,**kwargs):
        '''
        Multiplicative updates for training supervised NMF model (2).

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
            If saveerrs, returns array of ||X - AS||_F^2 + lam ||Y - BS||_F^2 for each
            iteration (length numiters).
        reconerrs : array, optional
            If saveerrs, returns array of ||X - AS||_F for each iteration (length numiters).
        classerrs : array, optional
            If saveerrs, returns array of ||Y - BS||_F for each iteration (length numiters).
        classaccs : array, optional
            If saveerrs, returns array of classification accuracy (computed with Y, B, S) at each
            iteration (length numiters).
        '''
        numiters = kwargs.get('numiters', 10)
        saveerrs = kwargs.get('saveerrs', False)
        eps = kwargs.get('eps', 1e-10)
    
    
        if saveerrs:
            errs = np.empty(numiters) #initialize error array
            reconerrs = np.empty(numiters)
            classerrs = np.empty(numiters)
            classaccs = np.empty(numiters)
        
        if self.Y is None:
            #if no label matrix provided, train unsupervised model instead
            raise Exception('Label matrix Y not provided: train with mult instead.')
    
        for i in range(numiters):
            #multiplicative updates for A, S, and B
            self.A = np.multiply(np.divide(self.A,eps+ self.A @ self.S @ np.transpose(self.S)), \
                                 self.X @ np.transpose(self.S))
            self.B = np.multiply(np.divide(self.B, eps+ self.B @ self.S @ np.transpose(self.S)), \
                                 self.Y @ np.transpose(self.S))
            self.S = np.multiply(np.divide(self.S, eps+ np.transpose(self.A) @ self.A @ self.S + \
                                self.lam * np.transpose(self.B) @ self.B @ self.S), \
                                np.transpose(self.A) @ self.X + self.lam * np.transpose(self.B) \
                                @ self.Y)
        
            if saveerrs:
                reconerrs[i] = la.norm(self.X - self.A @ self.S, 'fro') 
                classerrs[i] = la.norm(self.Y - self.B @ self.S, 'fro')
                errs[i] = reconerrs[i]**2 + self.lam * classerrs[i]**2 #save errors
                classaccs[i] = self.accuracy()
        
        if saveerrs:
            return [errs,reconerrs,classerrs,classaccs]
        
    def klsnmfmult(self,**kwargs):
        '''
        Multiplicative updates for training supervised NMF model (3).

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
            If saveerrs, returns array of ||X - AS||_F^2 + lam D(Y||BS) for each
            iteration (length numiters).
        reconerrs : array, optional
            If saveerrs, returns array of ||X - AS||_F for each iteration (length numiters).
        classerrs : array, optional
            If saveerrs, returns array of D(Y||BS) for each iteration (length numiters).
        classaccs : array, optional
            If saveerrs, returns array of classification accuracy (computed with Y, B, S) at each
            iteration (length numiters).
        '''
        numiters = kwargs.get('numiters', 10)
        saveerrs = kwargs.get('saveerrs', False)
        eps = kwargs.get('eps', 1e-10)
    
    
        if saveerrs:
            errs = np.empty(numiters) #initialize error array
            reconerrs = np.empty(numiters)
            classerrs = np.empty(numiters)
            classaccs = np.empty(numiters)
        
        if self.Y is None:
            #if no label matrix provided, train unsupervised model instead
            raise Exception('Label matrix Y not provided: train with mult instead.')
                
        classes = np.shape(self.Y)[0]
        cols = np.shape(self.Y)[1]
    
        for i in range(numiters):
            #multiplicative updates for A, S, and B
            self.A = np.multiply(np.divide(self.A,eps+ self.A @ self.S @ np.transpose(self.S)), \
                                 self.X @ np.transpose(self.S))
            self.B = np.multiply(np.divide(self.B,eps+ np.ones((classes,cols)) @ np.transpose(self.S)), \
                                 np.divide(self.Y, eps+ self.B @ self.S) @ np.transpose(self.S))
            self.S = np.multiply(np.divide(self.S, eps+ 2 * np.transpose(self.A) @ self.A @ self.S + \
                                           self.lam * np.transpose(self.B) @ \
                                           np.ones((classes,cols))),2 * np.transpose(self.A) \
                                 @ self.X + self.lam * np.transpose(self.B) @ \
                                 np.divide(self.Y, eps+ self.B @ self.S))
        
            if saveerrs:
                reconerrs[i] = la.norm(self.X - self.A @ self.S, 'fro') 
                classerrs[i] = self.kldiv()
                errs[i] = reconerrs[i]**2 + self.lam * classerrs[i] #save errors
                classaccs[i] = self.accuracy()
        
        if saveerrs:
            return [errs,reconerrs,classerrs,classaccs]

    def accuracy(self,**kwargs):
        '''
        Compute accuracy of supervised model (2) or (3) above.

        Returns
        -------
        acc : float_
            Fraction of correctly classified data points (computed with Y, B, S).
        '''

        if self.Y is None:
            raise Exception('Label matrix Y not provided: model is not supervised.')
        
        numdata = np.shape(self.Y)[1]

        #count number of data points which are correctly classified
        numacc = 0
        Yhat = self.B @ self.S
        for i in range(numdata):
            true_max = np.argmax(self.Y[:,i])
            approx_max = np.argmax(Yhat[:,i])

            if true_max == approx_max:
                numacc = numacc + 1

        #return fraction of correctly classified data points
        acc = numacc/numdata
        return acc

    def kldiv(self,**kwargs):
        '''
        Compute KL-divergence between Y and BS of supervised model (most naturally (3)).

        Parameters
        ----------
        eps : float_, optional
            Epsilon value to prevent division by zero (default is 1e-10).

        Returns
        -------
        kldiv : float_
            KL-divergence between Y and BS.
        '''
        eps = kwargs.get('eps', 1e-10)

        if self.Y is None:
            raise Exception('Label matrix Y not provided: model is not supervised.')

        #compute divergence
        Yhat = self.B @ self.S
        div = np.multiply(self.Y, np.log(np.divide(self.Y+eps, Yhat+eps))) - self.Y + Yhat
        kldiv = np.sum(np.sum(div))
        return kldiv
