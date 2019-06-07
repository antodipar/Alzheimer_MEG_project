
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


def rf(X, Y, param=100):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=param, random_state=1)
    clf.fit(X, Y)
    return clf.feature_importances_


def lr(X, Y, param=1e-2):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=param, penalty='l2', random_state=1, class_weight='balanced')
    clf.fit(X, Y)
    return clf.coef_.squeeze()


def statiblity(X,Y):
    from sklearn.linear_model import RandomizedLogisticRegression

    clf = RandomizedLogisticRegression(random_state=1)
    clf.fit(X,Y)

    return clf.scores_


