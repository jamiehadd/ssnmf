\usepackage[hybrid]{markdown}
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

First, import the `ssnmf` package and the relevant class `SSNMF`.  We import `numpy` for experimentation. 

```python
>>> import ssnmf
>>> from ssnmf import SSNMF
>>> import numpy as np

```

#### Training an unsupervised model

Declare an unsupervised NMF model with data matrix `X` and number of topics `k`.  Run the multiplicative updates method for this unsupervised model for `N` iterations.  This method tries to minimize the objective function $\|X - AS\|_F^2$.

```python
>>> X = np.random.rand(100,100)
>>> k = 10
>>> model = SSNMF(X,k)
>>> N = 100
>>> model.mult(numiters = N)
```


## Citing
If you use our work in an academic setting, please cite our paper:



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
