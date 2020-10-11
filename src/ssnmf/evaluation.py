""" Class for evaluating (S)SNMF models. """


import numpy as np
from numpy import linalg as la
import scipy.optimize.nnls as nnls

import ssnmf.ssnmf


class Evaluation:
    """
    Class for evaluating (S)SNMF models on train and test data. It contains functions
    to compute the reconstruction errors and classifications accuracies for each model
    on the train and test data.

    Model Numbers
    ----------
    (3) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 or
    (4) ||X - AS||_F^2 + lam * D(Y||BS) or
    (5) D(X||AS) + lam * ||Y - BS||_F^2 or
    (6) D(X||AS) + lam * D(Y||BS)

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
                     Number indicating which of the (S)SNMF models to train (see model numbers above).
    A              : array, optional
                     Initialization for the left factor matrix of the data matrix, shape (#features, k)
                     (the default is a matrix with uniform random entries).
    S              : array, optional
                     Initialization for the right factor matrix of the train data/label matrix,
                     shape (k, #train samples), (the default is a matrix with uniform random entries).
    B              : array, optional
                     Initialization for left factor matrix of the train label matrix, shape (#classes, k),
                     (the default is a matrix with uniform random entries).
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
    tol            : float_, optional
                     tolerance for terminating the model (default is 1e-4).
    iter_s         : int_, optional
                     Number of iterations of updates to run to approximate the representation of the test
                     data when the I-divervene a discrepancy measure for data reconstruction (default is 10).


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


    Usage Example
    ----------
    import evaluation
    from evaluation import Evaluation

    # Run hyperparameter search (Optional)
    param_module = Evaluation(train_features = X_train,
                                 train_labels = y_train,
                                 test_features = X_val,
                                 test_labels = y_val,
                                 k = 14, modelNum = 4)
    [opt_lamb, opt_ka, opt_itas] = param_module.iterativeLocalSearch(init_lamb=100, init_k=14, init_itas=8)

    # Run model with "optimal" parameters.
    eval_module = Evaluation(train_features = np.concatenate((X_train, X_val), axis=1),
                                 train_labels = np.concatenate((y_train, y_val), axis=1),
                                 test_features = X_test,
                                 test_labels = y_test,
                                 modelNum = 4, k = opt_ka,
                                 lam=opt_lamb, numiters = opt_itas)

    train_evals, test_evals = eval_module.eval()

    print(f"The classification accuracy on the train data is {train_evals[-1][-1]*100:.2f}%")
    print(f"The classification accuracy on the test data is {test_evals[-1]*100:.2f}%")

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
        self.iter_s = kwargs.get('iter_s', 20)
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
            raise Exception('The row dimension W_train is not equal to the number of features in the data.')
        if np.shape(self.W_train)[1] != self.cols:
            raise Exception('The column dimension of W_train is not equal to the number of samples in the train and test data.')

        self.W_test = kwargs.get('W_test', np.ones(self.test_features.shape))
        if np.shape(self.W_test)[0] != self.rows:
            raise Exception('The row dimension W_test is not equal to the number of features in the data.')
        if np.shape(self.W_test)[1] != self.test_features.shape[1]:
            raise Exception('The column dimension of W_test is not equal to the number of samples in the test data.')

        self.L = kwargs.get('L', np.concatenate((np.ones(self.train_labels.shape),np.zeros(self.test_labels.shape)),axis=1))
        if np.shape(self.L)[0] != self.classes:
            raise Exception('The row dimension of L is not equal to the number of classes in the data.')
        if np.shape(self.L)[1] != self.cols:
            raise Exception('The column dimension of L is not equal to the number of samples in the train and test data.')

        self.model = ssnmf.SSNMF(X = np.concatenate((self.train_features, self.test_features), axis = 1), k = self.k, \
                                        modelNum = self.modelNum, Y = np.concatenate((self.train_labels, self.test_labels), axis = 1), \
                                        W = self.W_train, L = self.L, A = self.A, B = self.B, S = self.S, lam = self.lam, tol = self.tol)

    def eval(self):
        '''
        This function fits the (S)SNMF model to the train data, and evaluates the performance of the model on the train and test data.

        Returns:
            train_evals (list): [ndarray(total reconstruction error (model objective function) on train data for each iteration),
                                 ndarray(data reconstruction error on train data for each iteration),
                                 ndarray(label reconstruction error on train data for each iteration),
                                 ndarray(classification accuracy on train data for each iteration)]
            test_evals (list): [float(total reconstruction error (model objective function) on test data),
                                float(data reconstruction error on test data),
                                float(label reconstruction error on test data),
                                float(classification accuracy on test data)]
        '''
        # Fit (S)SNMF model to train data
        train_evals = self.model.mult(numiters = self.numiters, saveerrs = True) #save train data errors
        # Apply (S)SNMF model to test data
        self.S_test = self.test_model() #representation matrix of test data
        # Compute reconstruction errors on test data
        test_evals = self.computeErrors()

        return train_evals, test_evals


    def test_model(self):
        '''
        Given a trained (S)SNMF model i.e. learned data dictionary, A, the function applies the (S)SNMF model
        to the test data to produce the representation of the test data.

        Returns:
            S_test (ndarray): representation matrix of the test data, shape(#topics, #test features)
        '''
        if self.modelNum == 3 or self.modelNum == 4: # Frobenius discrepancy measure on features data (use nonnegative least squares)
            S_test = np.zeros([self.k, np.shape(self.test_features)[1]])
            for i in range(np.shape(self.test_features)[1]):
                S_test[:,i] = nnls(np.multiply(np.ones(self.model.A.shape) * self.W_test[:,i].reshape(-1,1),self.model.A), \
                                    np.multiply(self.W_test[:,i], self.test_features[:,i]))[0]

        if self.modelNum == 5 or self.modelNum == 6: # I-divergence discrepancy measure on features data (use mult. upd. I-div)
            S_test = np.random.rand(self.k, self.test_features.shape[1])
            for i in range(self.iter_s):
                S_test = np.transpose(self.model.dictupdateIdiv(np.transpose(self.test_features), np.transpose(S_test), \
                                       np.transpose(self.model.A), np.transpose(self.W_test), eps= 1e-10))
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

    def iterativeLocalSearch(self, init_lamb, init_k, init_itas):
        """
        This function allows the user to perform an iterative local search in hyperparameter space for
        a set of [locally] optimal (k,lambda,iterations). This is done by starting with a choice
        of rank (init_k), iterations (init_itas), and lambda (init_lamb). Then by performing a parameter
        sweep in a "local search cube", the algorithm can determine if a new choice of (k,lambda,iterations)
        is advantageous i.e. produces better classification accuracy on validation set.

        Several hard-coded choices were made to ease the user-required knowledge of the search algorithm itself.
        These are for example:
            - The size of the local search box is (k-1 , k+2) x (iter-1, iter+1) x (0.1*lam , 10*lam).
            - The number of iterations tested uniformly spaced in the interval (iter-1, iter+1), which currently is 2 values.
            - The number of lambdas tested uniformly spaced in the interval (0.1*lam, 10*lam), which currently is 3 values.

        Args:
            init_lamb (float): initialization of the regualrizer lambda
            init_k (int): initialization of the rank
            init_itas (int): initialization of the number of iterations

        Returns:
            lamb (float): optimal regualrizer lambda found in the local search
            ka (int): optimal number of topics found in the local search
            itas (int): optimal number of iterations found in the local search
        """

        self.lam = init_lamb
        self.k = init_k
        self.numiters = init_itas
        self.converged = False
        self.iteration = 1

        if self.k <2:
            raise Exception('The initialization of the number of topics should be at least 2.')

        if self.numiters <2:
            raise Exception('The initialization of the number of iterations should be at least 2.')

        # Hyper-hyperparameters
        self.max_search_iter = 10     # Maximum number of local parameter sweeps performed
        self.ww              = 0.001   # Threshold for convergence - continue as long as a 0.1% improvement in accuracy.

        # Get the initial error
        eval_module = Evaluation(train_features = self.train_features, train_labels = self.train_labels,\
                                test_features = self.test_features, test_labels = self.test_labels,\
                                modelNum = self.modelNum, k = self.k, lam=self.lam, numiters = self.numiters,\
                                W_train = self.W_train, W_test= self.W_test,\
                                L = self.L, tol = self.tol, iter_s = self.iter_s)

        train_evals, test_evals = eval_module.eval()
        self.cls_last = test_evals[-1]

        print(f"Initial hyperparameters: k = {self.k}, iter = {self.numiters}, and lam = {self.lam}.")
        print(f"Initial val total reconstruction error = {test_evals[0]}")
        print(f"Initial val data reconstruction error = {test_evals[1]}")
        print(f"Initial val classification error = {test_evals[2]}")
        print(f"Initial val classification accuracy: {test_evals[3]}")

        while(not self.converged):
            # Perform a local search on k, itas, and lam
            best_ka = init_k
            best_itas = init_itas
            best_lamb = init_lamb
            best_local_acc = 0

            for k in range(self.k-1,self.k+2):
                for it in range(self.numiters-1,self.numiters+2,1):
                    startl = float(0.1*self.lam)
                    endl = float(10*self.lam)
                    la_vals = np.concatenate((np.linspace(startl, endl, num=2), np.array([self.lam])), axis=0)
                    for la_idx in range(len(la_vals)):
                        la = la_vals[la_idx]
                        print(f"Currently testing k = {k}, iteration = {it}, and lambda = {la}")
                        eval_module = Evaluation(train_features = self.train_features,train_labels = self.train_labels,\
                                                test_features = self.test_features, test_labels = self.test_labels,\
                                                modelNum = self.modelNum, k = k, lam=la, numiters = it, \
                                                W_train = self.W_train, W_test= self.W_test,\
                                                L = self.L, tol = self.tol, iter_s = self.iter_s)
                        train_evals, test_evals = eval_module.eval()
                        if(test_evals[-1] > best_local_acc):
                            best_ka = k
                            best_itas = it
                            best_lamb = la
                            best_local_acc = test_evals[-1]

            # Recover the performance of the best (k,lamb,iter) in the local search box
            eval_module = Evaluation(train_features = self.train_features, train_labels = self.train_labels,\
                                    test_features = self.test_features, test_labels = self.test_labels,\
                                    modelNum = self.modelNum, k = best_ka, lam=best_lamb, numiters = best_itas,\
                                    W_train = self.W_train, W_test= self.W_test,\
                                    L = self.L, tol = self.tol, iter_s = self.iter_s)
            train_evals, test_evals = eval_module.eval()
            cls_current = test_evals[-1]

            # Compare the best performing (k,lamb,iter) with the previous best choice
            if(cls_current >= (1 + self.ww)*self.cls_last):
                self.converged  = False
                self.cls_last = cls_current
                self.k = best_ka
                self.numiters = best_itas
                self.lam = best_lamb
                self.iteration += 1

                if(self.iteration % 1 == 0):
                    print("Iteration:",self.iteration)
                    print("The new selection of hyperparameters: k =", self.k, "; iter =", self.numiters, "; lamb =", self.lam)
                    print(f"Current val total reconstruction error = {test_evals[0]}")
                    print(f"Current val data reconstruction error = {test_evals[1]}")
                    print(f"Current val classification (div) error = {test_evals[2]}")
                    print(f"Current val classification accuracy: {test_evals[3]}")

                if self.k  < 2:
                    self.converged = True
                    print('The search method stopped because the number of topics found is less than two.')
                    return
                if self.numiters  < 2:
                    self.converged = True
                    print('The search method stopped because the number of iterations found is less than two.')
                    return

            elif(self.iteration > self.max_search_iter):
                self.converged = True
                print("The maximum number of search iterations is reached. The set of hyperparameters found are:")
                print("k =", self.k, "; iter =", self.numiters, "; lamb =", self.lam)

            else:
                self.converged = True
                print("Converged!")
                print("The set of hyperparameters found at convergence were:")
                print("k =", self.k, "; iter =", self.numiters, "; lam =", self.lam)
                print(f"Final val classification accuracy: {self.cls_last}")


        return self.lam, self.k, self.numiters
