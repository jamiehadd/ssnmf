Run exp_20news.py with specified parameters to reproduce the paper's results.

- We use rank = 13 for all experiments.
- We use iterations = 11 for reporting all the classification results.

To reproduce the classification results set run_analysis = 1.
To reproduce the results for the selected nmf parameters set nmf_search = 1 (with iterations = 10).
To reproduce the results for the selected ssnmf parameters set ssnmf_search = 1 (with iterations = 10).
To reproduce the clustering results set clust_analysis = 1.

Note: The evaluation module of the ssnmf package uses the nonnegative least squared (nnls) method from scipy.optimize. To avoid running into issues, please install scipy version 1.4.1 (pip install 'scipy==1.4.1').