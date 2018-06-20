

import numpy as np

def reconstruct_net(links, nROIs):

    W = np.zeros((nROIs, nROIs))
    mask = np.triu(np.ones((nROIs, nROIs), dtype=bool), k=1)
    W[mask] = links
    return W + W.T


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

    for iden, density in enumerate(densities):

        newlinks = links.copy()
        nlinks2keep = int(round(density*1.0/100*nlinks))
        # newlinks[ranking[:nlinks2keep]] = 1
        newlinks[ranking[nlinks2keep:]] = 0
        W = reconstruct_net(newlinks, nROIs)
        thr_nets[iden] = W


    return thr_nets

def compute_metrics(nets):

    import networkx as nx

    features = np.array([])

    for W in nets:

        W = nx.Graph(W)

        # --- strength
        st = W.degree(weight = 'weight')
        strength = np.array([info[1] for info in st])
        strength=(strength-strength.mean())/strength.std(ddof=1)
        # --- closeness
        closeness = np.array(nx.closeness_centrality(W, distance = 'weight').values())
        closeness = (closeness-closeness.mean())/closeness.std(ddof=1)
        # --- betweenness centrality
        betweenness = np.array(nx.betweenness_centrality(W, weight = 'weight').values())
        betweenness = (betweenness-betweenness.mean())/betweenness.std(ddof=1)
        # --- eigenvector
        eigenvector = np.array(nx.eigenvector_centrality(W, weight = 'weight').values())
        eigenvector = (eigenvector-eigenvector.mean())/eigenvector.std(ddof=1)

        # --- harmonic
        harmonic = np.array(nx.harmonic_centrality(W).values())
        harmonic = (harmonic-harmonic.mean())/harmonic.std(ddof=1)

        # # --- clustering
        clustering = np.array(nx.clustering(W, weight = 'weight').values())
        clustering=(clustering-clustering.mean())/clustering.std(ddof=1)
        # ---concatenate features
        features = np.concatenate((features, strength, closeness, eigenvector, betweenness, clustering, harmonic))

    return features



