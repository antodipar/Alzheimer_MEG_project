
"""
This module implements different techniques to quantify
the relative importance of each feature.
"""

import numpy as np

def ttest(X, Y):
    # X: features, Y: labels

    from scipy import stats
    nfeats = X.shape[1]
    condA = np.where(Y == False)[0]
    condB = np.where(Y == True)[0]

    importance = np.zeros((nfeats))

    for i in np.arange(nfeats):
        t,p = stats.ttest_ind(X[condA, i], X[condB, i], equal_var=False)
        importance[i] = 1-p

    return importance


def wilcoxon(X, Y):
    # X: features, Y: labels

    from scipy import stats
    nfeats = X.shape[1]
    condA = np.where(Y == False)[0]
    condB = np.where(Y == True)[0]

    importance = np.zeros((nfeats))

    for i in np.arange(nfeats):
        s,p = stats.ranksums(X[condA, i], X[condB, i])
        importance[i] = 1-p

    return importance


