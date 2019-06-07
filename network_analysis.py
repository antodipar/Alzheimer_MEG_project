

import numpy as np

def reconstruct_net(links, nROIs):

    W = np.zeros((nROIs, nROIs))
    mask = np.triu(np.ones((nROIs, nROIs), dtype=bool), k=1)
    W[mask] = links
    return W + W.T


def thresholding(W, density):

    # initialize some variables
    nROIs= W.shape[0]
    mask = np.triu(np.ones((nROIs, nROIs), dtype=bool), k=1)
    links = W[mask]
    ranking = np.argsort(links)
    ranking = ranking[::-1]
    nlinks = len(links)
    nlinks2keep = int(round(density * 1.0 / 100 * nlinks))
    # links[ranking[:nlinks2keep]] = 1
    links[ranking[nlinks2keep:]] = 0
    return reconstruct_net(links, nROIs)


def compute_metrics(W):

    import networkx as nx

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

    # # --- clustering
    clustering = np.array(nx.clustering(W, weight = 'weight').values())
    clustering=(clustering-clustering.mean())/clustering.std(ddof=1)

    # ## --- pagerank
    # pagerank = np.array(nx.pagerank(W, weight='weight').values())
    # pagerank = (pagerank - pagerank.mean()) / pagerank.std(ddof=1)
    #
    # ## --- hits
    # hits = np.array(nx.hits(W)[0].values())
    # hits = (hits - hits.mean()) / hits.std(ddof=1)


    # ---concatenate features
    return np.hstack((strength, closeness, betweenness, eigenvector, clustering))


