

import numpy as np


def thresholding(net, densities):

    # initialize some variables
    nROIs= net.shape[0]
    mask = np.triu(np.ones((nROIs, nROIs), dtype=bool), k=1)
    links = net[mask]
    ranking = np.argsort(links)
    ranking = ranking[::-1]
    nlinks = len(links)
    # the outcome to be returned
    thr_nets = np.zeros((len(densities), nROIs, nROIs))

    for i, W in enumerate(thr_nets):
        density = densities[i]
        nlinks2keep = int(round(density*1.0/100*nlinks))
        newlinks = links.copy()
        newlinks[ranking[:nlinks2keep]] = 1
        newlinks[ranking[nlinks2keep:]] = 0
        W[mask] = newlinks
        thr_nets[i] = W + W.T


    return thr_nets

def compute_metrics(nets):

    import networkx as nx

    features = np.array([])
    nROIs = nets.shape[1]
    for W in nets:

        W = nx.Graph(W)

        # --- strength
        st = W.degree(weight = 'weight')
        strength = np.zeros(nROIs)
        for info in st:
            strength[info[0]] = info[1]

        # --- closeness
        closeness = np.array(nx.closeness_centrality(W, distance = 'weight').values())

        # --- eigenvector
        eigenvector = np.array(nx.eigenvector_centrality(W, weight = 'weight').values())


        # --- clustering
        clustering = np.array(nx.clustering(W, weight = 'weight').values())

        # ---concatenate features
        features = np.concatenate((features, strength, closeness, eigenvector, clustering))

    return features

