import numpy as np
from numpy import linalg as la
from scipy.io import loadmat

class SSNMF:
    '''
    Class for (S)NMF model.
    
    The NMF model consists of the data matrix to be factorized, X, the factor matrices, A and S.  Each model also
    consists of a label matrix, Y, classification factor matrix, B, and classification weight parameter, lam
    (although these three variables will be empty if Y is not input).  These parameters define the objective function
    defining the model: 
    (1) ||X - AS||_F^2 (train with mult) or 
    (2) ||X - AS||_F^2 + lam * ||Y - BS||_F^2 (train with snmfmult) or 
    (3) ||X - AS||_F^2 + lam * D(Y||BS) (train with klsnmfmult).

    Examples
    --------
    >>> #unsupervised (1), saving errors
    >>> numIters = 100
    >>> model = SSNMF(numpy.random.rand(100,100),10)
    >>> errs = model.mult(saveerrs = True,numiters = numIters)
    
    >>> #unsupervised (1), not saving errors
    >>> numIters = 100
    >>> model = SSNMF(numpy.random.rand(100,100),10)
    >>> model.mult(numiters = numIters)
    
    >>> #supervised (2), saving errors
    >>> model = SSNMF(data['datamat'], 10, Y = data['labelmat'])
    >>> errs = model.snmfmult(numiters = numIters, saveerrs = True)
    
    >>> #supervised (3), saving errors
    >>> model = SSNMF(data['datamat'], 10, Y = data['labelmat'])
    >>> errs = model.klsnmfmult(numiters = numIters, saveerrs = True)
    '''
    def __init__(self, X, k, **kwargs):
        self.X = X
        rows = np.shape(X)[0]
        cols = np.shape(X)[1]
        self.A = kwargs.get('A',np.random.rand(rows,k)) #initialize factor A
        self.S = kwargs.get('S',np.random.rand(k,cols)) #initialize factor S
        
        #supervision initializations (optional)
        self.Y = kwargs.get('Y',None)
        if self.Y is not None:
            classes = np.shape(self.Y)[0]
            self.B = kwargs.get('B',np.random.rand(classes,k))
            self.lam = kwargs.get('lam',1)
        else:
            self.B = None
            self.lam = None
                       
    def mult(self,**kwargs):
        '''
        Multiplicative updates for training unsupervised NMF model.  This is for model (1) above.
        '''
        numiters = kwargs.get('numiters', 10)
        saveerrs = kwargs.get('saveerrs', False)
        
        if saveerrs:
            errs = np.empty(numiters) #initialize error array 
    
        for i in range(numiters):
            #multiplicative updates for A and S
            self.A = np.multiply(np.divide(self.A,self.A @ self.S @ np.transpose(self.S)), self.X @ np.transpose(self.S))
            self.S = np.multiply(np.divide(self.S,np.transpose(self.A) @ self.A @ self.S), np.transpose(self.A) @ self.X)
        
            if saveerrs:
                errs[i] = la.norm(self.X - self.A @ self.S, 'fro') #save reconstruction error
        
        if saveerrs:
            return [errs]
        
    def snmfmult(self,**kwargs):
        '''
        Multiplicative updates for training supervised NMF model.  This is for model (2) above.
        '''
        numiters = kwargs.get('numiters', 10)
        saveerrs = kwargs.get('saveerrs', False)
    
    
        if saveerrs:
            errs = np.empty(numiters) #initialize error array
            reconerrs = np.empty(numiters)
            classerrs = np.empty(numiters)
        
        if self.Y is not None:
            #if no label matrix provided, train unsupervised model instead
            print('Label matrix Y not provided: running NMF multiplicative updates instead.')
            if saveerrs:
                errs = self.mult(numiters = numiters, saveerrs = saveerrs)
                return [errs, reconerrs,classerrs]
            else:
                self.mult(numiters = numiters, saveerrs = saveerrs)
                return
    
        for i in range(numiters):
            #multiplicative updates for A, S, and B
            self.A = np.multiply(np.divide(self.A,self.A @ self.S @ np.transpose(self.S)), self.X @ np.transpose(self.S))
            self.B = np.multiply(np.divide(self.B, self.B @ self.S @ np.transpose(self.S)), self.Y @ np.transpose(self.S))
            self.S = np.multiply(np.divide(self.S, np.transpose(self.A) @ self.A @ self.S + self.lam * np.transpose(self.B) @ self.B @ self.S), np.transpose(self.A) @ self.X + self.lam * np.transpose(self.B) @ self.Y)
        
            if saveerrs:
                errs[i] = la.norm(self.X - self.A @ self.S, 'fro') + self.lam * la.norm(self.Y - self.B @ self.S, 'fro') #save errors
                reconerrs[i] = la.norm(self.X - self.A @ self.S, 'fro') 
                classerrs[i] = la.norm(self.Y - self.B @ self.S, 'fro')
        
        if saveerrs:
            return [errs,reconerrs,classerrs]
        
    def klsnmfmult(self,**kwargs):
        '''
        Multiplicative updates for training supervised NMF model.  This is for model (3) above.
        '''
        numiters = kwargs.get('numiters', 10)
        saveerrs = kwargs.get('saveerrs', False)
    
    
        if saveerrs:
            errs = np.empty(numiters) #initialize error array
            reconerrs = np.empty(numiters)
            classerrs = np.empty(numiters)
        
        if self.Y is not None:
            #if no label matrix provided, train unsupervised model instead
            print('Label matrix Y not provided: running NMF multiplicative updates instead.')
            if saveerrs:
                errs = self.mult(numiters = numiters, saveerrs = saveerrs)
                return [errs, reconerrs,classerrs]
            else:
                self.mult(numiters = numiters, saveerrs = saveerrs)
                return
                
        classes = np.shape(self.Y)[0]
        cols = np.shape(self.Y)[1]
    
        for i in range(numiters):
            #multiplicative updates for A, S, and B
            self.A = np.multiply(np.divide(self.A,self.A @ self.S @ np.transpose(self.S)), self.X @ np.transpose(self.S))
            self.B = np.multiply(np.divide(self.B,np.ones((classes,cols)) @ np.transpose(self.S)), np.divide(self.Y, self.B @ self.S) @ np.transpose(self.S))
            self.S = np.multiply(np.divide(self.S, 2 * np.transpose(self.A) @ self.A @ self.S + self.lam * np.transpose(self.B) @ np.ones((classes,cols))),2 * np.transpose(self.A) @ self.X + self.lam * np.transpose(self.B) @ np.divide(self.Y, self.B @ self.S))
        
            if saveerrs:
                errs[i] = la.norm(self.X - self.A @ self.S, 'fro') + self.lam * la.norm(self.Y - self.B @ self.S, 'fro') #save errors
                reconerrs[i] = la.norm(self.X - self.A @ self.S, 'fro') 
                classerrs[i] = la.norm(self.Y - self.B @ self.S, 'fro')
        
        if saveerrs:
            return [errs,reconerrs,classerrs]
            
