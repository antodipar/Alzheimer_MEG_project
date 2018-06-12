
"""
This module implements different techniques to quantify
the relative importance of each feature.
"""

import numpy as np

def ttest(X, Y):
    # Feature importance by comparing the means between class labels of each individual feature
    # X: features, Y: labels

    from scipy import stats
    nfeats = X.shape[1]
    condA = np.where(Y == False)[0]
    condB = np.where(Y == True)[0]

    importance = np.zeros((nfeats))

    for i in np.arange(nfeats):
        t,p = stats.ttest_ind(X[condA, i], X[condB, i], equal_var=False)
        importance[i] = abs(t)

    return importance