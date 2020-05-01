# SSNMF

[![PyPI Version](https://img.shields.io/pypi/v/ssnmf.svg)](https://pypi.org/project/ssnmf/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/ssnmf.svg)](https://pypi.org/project/ssnmf/)

SSNMF contains class for (SS)NMF model and several multiplicative update methods to train different models.

---

## Installation

To install SSNMF, run this command in your terminal:

```bash
    $ pip install -U ssnmf
```

This is the preferred method to install SSNMF, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## Usage

First, import the `ssnmf` package and the relevant class `SSNMF`.  We import `numpy` and `scipy' for experimentation. 

```python
>>> import ssnmf
>>> from ssnmf import SSNMF
>>> import numpy as np
>>> import scipy
>>> import scipy.sparse as sparse
>>> import scipy.optimize
```

#### Training an unsupervised model

Declare an unsupervised NMF model with data matrix `X` and number of topics `k`.  

```python
>>> X = np.random.rand(100,100)
>>> k = 10
>>> model = SSNMF(X,k)
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error ||X-AS||_F/||X||_F.

```python
>>> rel_error = np.linalg.norm(model.X - model.A @ model.S, 'fro')/np.linalg.norm(model.X,'fro')
```

Run the multiplicative updates method for this unsupervised model for `N` iterations.  This method tries to minimize the objective function `||X-AS||_F`. 

```python
>>> N = 100
>>> model.mult(numiters = N)
```

This method updates the factor matrices N times.  You can see how much the relative reconstruction error improves.

```python
>>> rel_error = np.linalg.norm(model.X - model.A @ model.S, 'fro')/np.linalg.norm(model.X,'fro')
```

#### Training a supervised model

We begin by generating some synthetic data for testing.
```python
>>> labelmat = np.concatenate((np.concatenate((np.ones([1,10]),np.zeros([1,30])),axis=1),np.concatenate((np.zeros([1,10]),np.ones([1,10]),np.zeros([1,20])),axis=1),np.concatenate((np.zeros([1,20]),np.ones([1,10]),np.zeros([1,10])),axis=1),np.concatenate((np.zeros([1,30]),np.ones([1,10])),axis=1)))
>>> B = sparse.random(4,10,density=0.2).toarray()
>>> S = np.zeros([10,40])
>>> for i in range(40):
	S[:,i] = scipy.optimize.nnls(B,labelmat[:,i])[0]
>>> A = np.random.rand(40,10)
>>> X = A @ S
```

Declare a supervised NMF model with data matrix `X`, number of topics `k`, label matrix `Y`, and weight parameter `lam`.  

```python
>>> k = 10
>>> model = SSNMF(X,k,Y = labelmat,lam=100*np.linalg.norm(X,'fro'))
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error ||X-AS||_F/||X||_F and classification accuracy.

```python
>>> rel_error = np.linalg.norm(model.X - model.A @ model.S, 'fro')/np.linalg.norm(model.X,'fro')
>>> acc = model.accuracy()
```

Run the multiplicative updates method for this supervised model for `N` iterations.  This method tries to minimize the objective function `||X-AS||_F^2 + lam ||Y - BS||_F^2`. This also saves the errors and accuracies in each iteration.

```python
>>> N = 100
>>> [errs,reconerrs,classerrs,classaccs] = model.snmfmult(numiters = N,saveerrs = True)
```

This method updates the factor matrices N times.  You can see how much the relative reconstruction error and classification accuracy improves.

```python
>>> rel_error = reconerrs[99]/np.linalg.norm(X,'fro')
>>> acc = classaccs[99]
```

#### Training a supervised model with KL-divergence

We begin by generating some synthetic data for testing.
```python
>>> labelmat = np.concatenate((np.concatenate((np.ones([1,10]),np.zeros([1,30])),axis=1),np.concatenate((np.zeros([1,10]),np.ones([1,10]),np.zeros([1,20])),axis=1),np.concatenate((np.zeros([1,20]),np.ones([1,10]),np.zeros([1,10])),axis=1),np.concatenate((np.zeros([1,30]),np.ones([1,10])),axis=1)))
>>> B = sparse.random(4,10,density=0.2).toarray()
>>> S = np.zeros([10,40])
>>> for i in range(40):
	S[:,i] = scipy.optimize.nnls(B,labelmat[:,i])[0]
>>> A = np.random.rand(40,10)
>>> X = A @ S
```

Declare a supervised NMF model with data matrix `X`, number of topics `k`, label matrix `Y`, and weight parameter `lam`.  

```python
>>> k = 10
>>> model = SSNMF(X,k,Y = labelmat,lam=100*np.linalg.norm(X,'fro'))
```

You may access the factor matrices initialized in the model, e.g., to check relative reconstruction error ||X-AS||_F/||X||_F, classification accuracy, and KL-divergence improves.

```python
>>> rel_error = np.linalg.norm(model.X - model.A @ model.S, 'fro')/np.linalg.norm(model.X,'fro')
>>> acc = model.accuracy()
>>> div = model.kldiv()
```

Run the multiplicative updates method for this supervised model for `N` iterations.  This method tries to minimize the objective function `||X-AS||_F^2 + lam D(Y||BS)`. This also saves the errors and accuracies in each iteration.

```python
>>> N = 100
>>> [errs,reconerrs,classerrs,classaccs] = model.klsnmfmult(numiters = N,saveerrs = True)
```

This method updates the factor matrices N times.  You can see how much the relative reconstruction error and classification accuracy improves.

```python
>>> rel_error = reconerrs[99]/np.linalg.norm(X,'fro')
>>> acc = classaccs[99]
>>> div = classerrs[99]
```


## Citing
If you use our code in an academic setting, please consider citing our code.
<!---Please cite our paper: ... -->



## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

#### Suggested Git Branch Strategy
1. `master` is for the most up-to-date development, very rarely should you directly commit to this branch. Your day-to-day work should exist on branches separate from `master`. It is recommended to commit to development branches and make pull requests to master.4. It is recommended to use "Squash and Merge" commits when committing PR's. It makes each set of changes to `master`
atomic and as a side effect naturally encourages small well defined PR's.


#### Additional Optional Setup Steps:
* Create an initial release to test.PyPI and PyPI.
    * Follow [This PyPA tutorial](https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives), starting from the "Generating distribution archives" section.

* Create a blank github repository (without a README or .gitignore) and push the code to it.

* Delete these setup instructions from `README.md` when you are finished with them.
