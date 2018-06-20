
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



def maxcc(X, Y, th = 1.5):
    # X: features, Y: labels

    import network_analysis as netanalysis
    reload(netanalysis)
    import networkx as nx
    from scipy import stats
    nfeats = X.shape[1]
    condA = np.where(Y == False)[0]
    condB = np.where(Y == True)[0]

    t_test = np.zeros((nfeats))

    for i in np.arange(nfeats):
        t, _ = stats.ttest_ind(X[condA, i], X[condB, i], equal_var=True)
        t_test[i] = abs(t)


    # reconstruct the network
    nROIs = 102
    ADJ = netanalysis.reconstruct_net(t_test, nROIs)
    ADJ = ADJ >= th

    # Find network components
    G = nx.from_numpy_matrix(ADJ)
    # Return connected components as subgraphs.
    comp_list = list(nx.connected_component_subgraphs(G))

    # store the number of edges for each subgraph component
    nr_edges_per_component = np.zeros(len(comp_list))

    for idx, componentG in enumerate(comp_list):
        nr_edges_per_component[idx] = componentG.number_of_edges()

    G = comp_list[np.argmax(nr_edges_per_component)]
    print "Max size: {}".format(nr_edges_per_component[np.argmax(nr_edges_per_component)])
    newADJ = np.zeros((nROIs, nROIs), dtype=bool)
    for ed in G.edges():
        newADJ[ed[0], ed[1]] = True

    newADJ = newADJ + newADJ.T


    return newADJ


def statiblity(X,Y):

    from sklearn.linear_model import RandomizedLogisticRegression

    clf = RandomizedLogisticRegression(random_state=1)
    clf.fit(X,Y)

    return clf.scores_


