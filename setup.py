#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

test_requirements = [
    "numpy",
    "torch"
    "scipy==1.4.1",
    "pytest",
]

docs_requirements = [
    
]

setup_requirements = [
    
]

dev_requirements = [
    *test_requirements,
    *docs_requirements,
    *setup_requirements,
    
    "bump2version>=1.0.3",
    "ipython>=7.5.0",
    
    "twine>=1.13.0",
    "wheel>=0.33.1",

    "tox>=3.15.0",
]

requirements = [
    "numpy",
    "torch",
]

extra_requirements = {
    "test": test_requirements,
    "docs": docs_requirements,
    "setup": setup_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *test_requirements,
        *docs_requirements,
        *setup_requirements,
        *dev_requirements,
    ],
}

setup(
    author="Jamie Haddock",
    author_email="jhaddock@math.ucla.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research ",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    description="SSNMF contains class for (SS)NMF model and several multiplicative update methods to train different models.",
    install_requires=requirements,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="ssnmf",
    name="ssnmf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.5",
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/jamiehadd/ssnmf",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="1.0.3",
    zip_safe=False,
)
