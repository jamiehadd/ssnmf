# SSNMF

[![PyPI Version](https://img.shields.io/pypi/v/ssnmf.svg)](https://pypi.org/project/ssnmf/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/ssnmf.svg)](https://pypi.org/project/ssnmf/)

SSNMF contains class for (SS)NMF model and several multiplicative update methods to train different models.

---

## Documentation

The NMF model consists of the data matrix to be factorized, X, the factor matrices, A and
S.  Each model also consists of a label matrix, Y, classification factor matrix, B, and
classification weight parameter, lam (although these three variables will be empty if Y is not
input).  These parameters define the objective function defining the model:
1. ||X - AS||<sub>F</sub><sup>2</sup>
2. D(X||AS) 
3. ||X - AS||<sub>F</sub><sup>2</sup> + &lambda;* ||Y - BS||<sub>F</sub><sup>2</sup>
4. ||X - AS||<sub>F</sub><sup>2</sup> + &lambda; * D(Y||BS) 
5. D(X||AS) + &lambda;* ||Y - BS||<sub>F</sub><sup>2</sup>
6. D(X||AS) + &lambda;* D(Y||BS)

+ Parameters

  + X        : numpy array or torch.Tensor
                Data matrix of size m x n.
  + k        : int_
                Number of topics.
  + modelNum : int_, optional<br>
                Number indicating which of above models user intends to train (the default is 1).
  + A        : numpy array or torch.Tensor, optional<br>
                Initialization for left factor matrix of X of size m x k (the default is a matrix with
                uniform random entries).
  + S        : numpy array or torch.Tensor, optional<br>
                Initialization for right factor matrix of X of size k x n (the default is a matrix with
                uniform random entries).
  + Y        : numpy array or torch.Tensor, optional<br>
                Label matrix of size p x n (default is None).
  + B        : numpy array or torch.Tensor, optional<br>
                Initialization for left factor matrix of Y of size p x k (the default is a matrix with
                uniform random entries if Y is not None, None otherwise).
  + lam      : float_, optional<br>
                Weight parameter for classification term in objective (the default is 1 if Y is not
                None, None otherwise).
  + W        : numpy array or torch.Tensor, optional<br>
                Missing data indicator matrix of same size as X (the defaults is matrix of all ones).
  + L        : numpy array or torch.Tensor, optional<br>
                Missing label indicator matrix of same size as Y (the default is matrix of all ones if
                Y is not None, None otherwise).
  + tol      : float_, optional<br>
                Tolerance for relative error stopping criterion (i.e., method stops when difference between consecutive relative errors falls below top)
  + str      : string, private<br>
               a flag to indicate whether this model is initialized by Numpy array or PyTorch tensor

+ Methods
  + mult(numiters = 10, saveerrs = True) <br>
    Train the selected model via numiters multiplicative updates
  + accuracy()<br>
    Compute the classification accuracy of supervised model (using Y, B, and S)
  + fronorm(Z, D, S, M) <br>
    Compute Frobenius norm ||Z - DS||<sub>F</sub><br>
    M is missing data indicator matrix of same size as Z (the defaults is matrix of all ones)
  + Idiv(Z, D, S, M) <br>
    Compute I-divergence D(Z||DS) <br>
    M is missing data indicator matrix of same size as Z (the defaults is matrix of all ones)




## Installation

To install SSNMF, run this command in your terminal:

```bash
    $ pip install -U ssnmf
```

This is the preferred method to install SSNMF, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## Usage

First, import the `ssnmf` package and the relevant class `SSNMF`.  We import `numpy`, `scipy` , and `torch` for experimentation. 
```python
>>> import ssnmf
>>> from ssnmf import SSNMF
>>> import numpy as np
>>> import torch
>>> import scipy
>>> import scipy.sparse as sparse
>>> import scipy.optimize
```

SSNMF can take both Numpy array and PyTorch Tensor to initialize an (SS)NMF model. 

If a model is initialized with PyTorch Tensor, `GPU` may be utilized to run the model.
To use `GPU` to run (SS)NMF, users should have `PyTorch` installed on their devices. To test if the `GPU` is available for their devices, run the following code. If it returns `True`, then `GPU` will be used to run this model, otherwise the CPU will be used.

```python
>>> torch.cuda.is_available()
```


### 1. Training an unsupervised model without missing data using Numpy array. 

Declare an unsupervised NMF model ||X - AS||<sub>F</sub><sup>2</sup> with data matrix `X` and number of topics `k`.  


```python
>>> X = np.random.rand(100,100)
>>> k = 10
>>> model = SSNMF(X,k,modelNum=1)
>>> A0 = model.A
>>> S0 = model.S
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error 
<span style="color:darkred;">||X - AS||<sub>F</sub><sup>2</sup> / ||X - A<sub>0</sub>S<sub>0</sub>||<sub>F</sub><sup>2</sup></span>

```python
>>> rel_error = model.fronorm(model.X, model.A, model.S, model.W)**2/model.fronorm(model.X, A0, S0, model.W)**2
>>> print("the initial relative reconstruction error is ", rel_error)
```

Run the multiplicative updates method for this unsupervised model for `N` iterations.  This method tries to minimize the objective function <span style="color:darkred;">||X - AS||<sub>F</sub></span>

```python
>>> N = 100
>>> [errs] = model.mult(numiters = N, saveerrs = True)
```

This method tries to updates the factor matrices N times. The actual number of updates depends on both N and the tolerance. You can see how many iterations that the model actually ran and how much the relative reconstruction error improves. 

```python
>>> size = errs.shape[0] 
>>> print("number of iterations that this model runs: ", size)
>>> rel_error = errs[size - 1]**2/model.fronorm(model.X, A0, S0, model.W)**2
>>> print("the final relative reconstruction  error is ", rel_error)
```

### 2. Training an unsupervised model without missing data using PyTorch tensor. 

Declare an unsupervised NMF model <span style="color:darkred;">D(X||AS)</span>
 with data matrix `X` and number of topics `k`.  

```python
>>> d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> X = torch.rand(100, 100, dtype=torch.float, device=d)
>>> k = 10
>>> model = SSNMF(X,k,modelNum=2)
>>> A0 = model.A
>>> S0 = model.S
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error 
<span style="color:darkred;">D(X||AS)/D(X||A<sub>0</sub>S<sub>0</sub>)</span>

```python
>>> rel_error = model.Idiv(model.X, model.A, model.S, model.W)/model.Idiv(model.X, A0, S0, model.W)
>>> print("the initial relative reconstruction  error is ", rel_error)
```

Run the multiplicative updates method for this unsupervised model for `N` iterations.  This method tries to minimize the objective function <span style="color:darkred;">D(X||AS)</span>

```python
>>> N = 100
>>> [errs] = model.mult(numiters = N, saveerrs = True)
```

This method tries to updates the factor matrices N times. The actual number of updates depends on both N and the tolerance. You can see how many iterations that the model actually ran and how much the relative reconstruction error improves. 

```python
>>> size = errs.shape[0] 
>>> print("number of iterations that this model runs: ", size)
>>> rel_error = errs[size - 1]/model.Idiv(model.X, A0, S0, model.W)
>>> print("the final relative reconstruction error is ", rel_error)
```

### 3. Training a supervised model without missing data using Numpy array

We begin by generating some synthetic data for testing.
```python
>>> labelmat = np.concatenate((np.concatenate((np.ones([1,10]),np.zeros([1,30])),axis=1),np.concatenate((np.zeros([1,10]),np.ones([1,10]),np.zeros([1,20])),axis=1),np.concatenate((np.zeros([1,20]),np.ones([1,10]),np.zeros([1,10])),axis=1),np.concatenate((np.zeros([1,30]),np.ones([1,10])),axis=1)))
>>> B = sparse.random(4,10,density=0.2).toarray()
>>> S = np.zeros([10,40])
>>> for i in range(40):
... 	S[:,i] = scipy.optimize.nnls(B,labelmat[:,i])[0]
>>> A = np.random.rand(40,10)
>>> X = A @ S
```

Declare a supervised SSNMF model <span style="color:darkred;">||X - AS||<sub>F</sub><sup>2</sup> + &lambda;* ||Y - BS||<sub>F</sub><sup>2</sup></span> with data matrix `X`, number of topics `k`, label matrix `Y`, and weight parameter <span style="color:darkred;">&lambda;</sup>.  

```python
>>> k = 10
>>> model = SSNMF(X,k,Y = labelmat,lam=100*np.linalg.norm(X,'fro'),modelNum=3)
>>> A0 = model.A
>>> S0 = model.S
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error <span style="color:darkred;">||X - AS||<sub>F</sub><sup>2</sup> / ||X - A<sub>0</sub>S<sub>0</sub>||<sub>F</sub><sup>2</sup></span> and classification accuracy.

```python
>>> rel_error = model.fronorm(model.X, model.A, model.S, model.W)**2/model.fronorm(model.X, A0, S0, model.W)**2
>>> acc = model.accuracy()
>>> print("the initial relative reconstruction error is ", rel_error)
>>> print("the initial classifier's accuracy is ", acc)
```

Run the multiplicative updates method for this supervised model for `N` iterations.  This method tries to minimize the objective function <span style="color:darkred;">||X - AS||<sub>F</sub><sup>2</sup> + &lambda;* ||Y - BS||<sub>F</sub><sup>2</sup></span> . This also saves the errors and accuracies in each iteration.

```python
>>> N = 100
>>> [errs,reconerrs,classerrs,classaccs] = model.mult(numiters = N,saveerrs = True)
```

This method updates the factor matrices N times.  You can see how much the relative reconstruction error and classification accuracy improves.

```python
>>> size = reconerrs.shape[0]
>>> rel_error = reconerrs[size - 1]**2/model.fronorm(model.X, A0, S0, model.W)**2
>>> acc = classaccs[size - 1]
>>> print("number of iterations that this model runs: ", size)
>>> print("the final relative reconstruction error is ", rel_error)
>>> print("the final classifier's accuracy is ", acc)
```

### 4. Training a supervised model without missing data using PyTorch tensor

Generating some synthetic data for testing.
```python
>>> labelmat = np.concatenate((np.concatenate((np.ones([1,10]),np.zeros([1,30])),axis=1),np.concatenate((np.zeros([1,10]),np.ones([1,10]),np.zeros([1,20])),axis=1),np.concatenate((np.zeros([1,20]),np.ones([1,10]),np.zeros([1,10])),axis=1),np.concatenate((np.zeros([1,30]),np.ones([1,10])),axis=1)))
>>> B = sparse.random(4,10,density=0.2).toarray()
>>> S = np.zeros([10,40])
>>> for i in range(40):
...     S[:,i] = scipy.optimize.nnls(B,labelmat[:,i])[0]
>>> A = np.random.rand(40,10)
>>> X = A @ S
```

Define a simple function to convert Numpy array to PyTorch tensor.<br>
parameter m : the numpy array to be converted to PyTorch tensor<br>
parameter device : device of the PyTorch tensor(e.g. GPU or CPU)<br>
(<span style="color:darkred;">Important notice </span>: When use the function torch.from_numpy() to convert numpy array to PyTorch tensor, the data may lose precision. Here we use it only because the data is artificially generated to ensure X can be decomposed to A and S. If you apply ssnmf model using PyTorch on your own real data, you should store the data as PyTorch tensors to avoid precision loss)
```python
>>> def getTensor(m, device):
>>>   mt = torch.from_numpy(copy.deepcopy(m))
>>>   mt = mt.type(torch.FloatTensor)
>>>   mt = mt.to(device)
>>>   return mt
```

Declare a supervised SSNMF model <span style="color:darkred;">||X - AS||<sub>F</sub><sup>2</sup> + &lambda;*D(Y||BS)</span> with data matrix `X`, number of topics `k`, label matrix `Y`, and weight parameter <span style="color:darkred;">&lambda;</sup>.  

```python
>>> devise = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> Xt = getTensor(X, devise)
>>> Yt = getTensor(labelmat, devise)
>>> k = 10
>>> model = SSNMF(Xt,k,Y = Yt,lam=100*torch.norm(Xt), modelNum=4)
>>> A0 = model.A
>>> S0 = model.S
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error <span style="color:darkred;">||X - AS||<sub>F</sub><sup>2</sup> / ||X - A<sub>0</sub>S<sub>0</sub>||<sub>F</sub><sup>2</sup></span> and classification accuracy.

```python
>>> rel_error = model.fronorm(model.X, model.A, model.S, model.W)**2/model.fronorm(model.X, A0, S0, model.W)**2
>>> acc = model.accuracy()
>>> print("the initial relative reconstruction error is ", rel_error)
>>> print("the initial classifier's accuracy is ", acc)
```

Run the multiplicative updates method for this supervised model for `N` iterations.  This method tries to minimize the objective function <span style="color:darkred;">||X - AS||<sub>F</sub><sup>2</sup> + &lambda;*D(Y||BS)</span>. This also saves the errors and accuracies in each iteration.

```python
>>> N = 100
>>> [errs,reconerrs,classerrs,classaccs] = model.mult(numiters = N,saveerrs = True)
```

This method updates the factor matrices N times.  You can see how much the relative reconstruction error and classification accuracy improves.

```python
>>> size = reconerrs.shape[0]
>>> rel_error = reconerrs[size - 1]**2/model.fronorm(model.X, A0, S0, model.W)**2
>>> acc = classaccs[size - 1]
>>> print("number of iterations that this model runs: ", size)
>>> print("the final relative reconstruction error is ", rel_error)
>>> print("the final classifier's accuracy is ", acc)
```

### 5. Training a supervised model with missing data using Numpy array

Generating some synthetic data for testing.
```python
>>> labelmat = np.concatenate((np.concatenate((np.ones([1,10]),np.zeros([1,30])),axis=1),np.concatenate((np.zeros([1,10]),np.ones([1,10]),np.zeros([1,20])),axis=1),np.concatenate((np.zeros([1,20]),np.ones([1,10]),np.zeros([1,10])),axis=1),np.concatenate((np.zeros([1,30]),np.ones([1,10])),axis=1)))
>>> B = sparse.random(4,10,density=0.2).toarray()
>>> S = np.zeros([10,40])
>>> for i in range(40):
... 	S[:,i] = scipy.optimize.nnls(B,labelmat[:,i])[0]
>>> A = np.random.rand(40,10)
>>> X = A @ S
```

Define a simple function to generate a W matirx(missing data indicator matrix ).<br>
parameter X : the matrix that with missing data<br>
parameter per : the percentage of missing data that X has(e.g. per=10 means 10% data of X is missing)<br>
(<span style="color:darkred;">Important notice: </span> this function is just for showing people how to use the ssnmf model when there is missing data in X. In practical application, use your own missing data indicator matrix based on your real data)
```python
>>> def getW(X, per):
>>>     num = round(per/100 * X.shape[0] * X.shape[1])
>>>     W = np.ones(shape = X.shape)
>>>     row = [i for i in range(X.shape[0])]
>>>     column = [i for i in range(X.shape[1])]
>>>     index = random.sample(list(itertools.product(row, column)), num)
>>>     for i in range(num):
>>>         W[index[i][0]][index[i][1]] = 0
>>>     return W
```

Declare a supervised SSNMF model <span style="color:darkred;">D(X||AS)<sub>F</sub><sup>2</sup> + &lambda;* ||Y - BS||<sub>F</sub><sup>2</sup></span> with data matrix `X`, number of topics `k`, label matrix `Y`, missing data indicator matrix `W`, and weight parameter <span style="color:darkred;">&lambda;</sup>.  

```python
>>> k = 10
>>> W0 = getW(X, 10)
>>> model = SSNMF(X,k,Y = labelmat,lam=100*np.linalg.norm(X,'fro'), W = W0, modelNum=5)
>>> A0 = model.A
>>> S0 = model.S
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error <span style="color:darkred;">D(X||AS)/D(X||A<sub>0</sub>S<sub>0</sub>)</span> and classification accuracy.

```python
>>> rel_error = model.Idiv(model.X, model.A, model.S, model.W)/model.Idiv(model.X, A0, S0, model.W)
>>> acc = model.accuracy()
>>> print("the initial relative reconstruction error is ", rel_error)
>>> print("the initial classifier's accuracy is ", acc)
```

Run the multiplicative updates method for this supervised model for `N` iterations.  This method tries to minimize the objective function <span style="color:darkred;">D(X||AS) + &lambda;* ||Y - BS||<sub>F</sub><sup>2</sup></span> . This also saves the errors and accuracies in each iteration.

```python
>>> N = 100
>>> [errs,reconerrs,classerrs,classaccs] = model.mult(numiters = N,saveerrs = True)
```

This method updates the factor matrices N times.  You can see how much the relative reconstruction error and classification accuracy improves.

```python
>>> size = reconerrs.shape[0]
>>> rel_error = reconerrs[size - 1]/model.Idiv(model.X, A0, S0, model.W)
>>> acc = classaccs[size - 1]
>>> print("number of iterations that this model runs: ", size)
>>> print("the final relative reconstruction error is ", rel_error)
>>> print("the final classifier's accuracy is ", acc)
```

### 6. Training a supervised model missing labels using PyTorch tensor

Generating some synthetic data for testing.
```python
>>> labelmat = np.concatenate((np.concatenate((np.ones([1,10]),np.zeros([1,30])),axis=1),np.concatenate((np.zeros([1,10]),np.ones([1,10]),np.zeros([1,20])),axis=1),np.concatenate((np.zeros([1,20]),np.ones([1,10]),np.zeros([1,10])),axis=1),np.concatenate((np.zeros([1,30]),np.ones([1,10])),axis=1)))
>>> B = sparse.random(4,10,density=0.2).toarray()
>>> S = np.zeros([10,40])
>>> for i in range(40):
...     S[:,i] = scipy.optimize.nnls(B,labelmat[:,i])[0]
>>> A = np.random.rand(40,10)
>>> X = A @ S
```

Define a simple function to generate an L matrix (Missing label indicator matrix)<br>
Parameter Y : the label matrix with missing data.<br>
Parameter per : the percentage of missing data that Y has(e.g. per=10 means 10% data of Y is missing) <br>
Parameter device: device of the PyTorch tensor(e.g. GPU or CPU) <br>
(<span style="color:darkred;">Important notice: </span> this function is just for showing people how to use the ssnmf model when there is missing data in label matrix Y. In practical application, use your own missing label indicator  matrix based on your real data)

```python    
>>> def getL(Y, per):
>>>     num = round(per/100 * Y.shape[1])
>>>     L = np.ones(shape = Y.shape)
>>>     column = [i for i in range(Y.shape[1])]
>>>     index = random.sample(column, num)
>>>     L[:,index] = 0
>>>     return L
```

Declare a supervised SSNMF model <span style="color:darkred;">D(X||AS) + &lambda;*D(Y||BS)</span> with data matrix `X`, number of topics `k`, label matrix `Y`, and weight parameter <span style="color:darkred;">&lambda;</sup>.  

```python
>>> devise = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> Xt = getTensor(X, devise) ## getTensor() defined in section 4.Training a supervised model without missing data using PyTorch tensor 
>>> Yt = getTensor(labelmat, devise)
>>> L0 = getL(Y, 10, device)
>>> k = 10
>>> model = SSNMF(Xt,k,Y = Yt,lam=100*torch.norm(Xt), L=L0, modelNum=6)
>>> A0 = model.A
>>> S0 = model.S
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error <span style="color:darkred;">D(X||AS)/D(X||A<sub>0</sub>S<sub>0</sub>)</span> and classification accuracy.

```python
>>> rel_error = model.Idiv(model.X, model.A, model.S, model.W)/model.Idiv(model.X, A0, S0, model.W)
>>> acc = model.accuracy()
>>> print("the initial relative reconstruction error is ", rel_error)
>>> print("the initial classifier's accuracy is ", acc)
```

Run the multiplicative updates method for this supervised model for `N` iterations.  This method tries to minimize the objective function <span style="color:darkred;">D(X||AS) + &lambda;*D(Y||BS)</span>. This also saves the errors and accuracies in each iteration.

```python
>>> N = 100
>>> [errs,reconerrs,classerrs,classaccs] = model.mult(numiters = N,saveerrs = True)
```

This method updates the factor matrices N times.  You can see how much the relative reconstruction error and classification accuracy improves.

```python
>>> size = reconerrs.shape[0]
>>> rel_error = reconerrs[size - 1]/model.Idiv(model.X, A0, S0, model.W)
>>> acc = classaccs[size - 1]
>>> print("number of iterations that this model runs: ", size)
>>> print("the final relative reconstruction error is ", rel_error)
>>> print("the final classifier's accuracy is ", acc)
```


## Citing
If you use our code in an academic setting, please consider citing the following paper.

J. Haddock, L. Kassab, S. Li, A. Kryshchenko, R. Grotheer, E. Sizikova, C. Wang, T. Merkh, R. W. M. A. Madushani, M. Ahn, D. Needell, and K. Leonard, "Semi-supervised Nonnegative Matrix Factorization Models for Topic Modeling in Learning Tasks." Submitted, 2020.
<!---Please cite our paper: ... -->



## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
