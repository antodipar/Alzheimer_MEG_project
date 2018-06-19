
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



def maxcc(X, Y, th = 3):
    # X: features, Y: labels

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
    ADJ = np.zeros((nROIs, nROIs))
    mask = np.triu(np.ones((nROIs, nROIs), dtype=bool), k=1)
    ADJ[mask] = t_test >= th
    ADJ = ADJ + ADJ.T

    # Find network components
    G = nx.from_numpy_matrix(ADJ)
    # Return connected components as subgraphs.
    comp_list = list(nx.connected_component_subgraphs(G))

    # store the number of edges for each subgraph component
    nr_edges_per_component = np.zeros(len(comp_list))

    for idx, componentG in enumerate(comp_list):
        nr_edges_per_component[idx] = componentG.number_of_edges()

    G = comp_list[np.argmax(nr_edges_per_component)]

    newADJ = np.zeros((nROIs, nROIs), dtype=bool)
    for ed in G.edges():
        newADJ[ed[0], ed[1]] = True

    newADJ = newADJ + newADJ.T


    return newADJ




