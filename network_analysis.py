

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

    # # --- harmonic
    # harmonic = np.array(nx.harmonic_centrality(W).values())
    # harmonic = (harmonic-harmonic.mean())/harmonic.std(ddof=1)

    # # --- clustering
    clustering = np.array(nx.clustering(W, weight = 'weight').values())
    clustering=(clustering-clustering.mean())/clustering.std(ddof=1)
    # ---concatenate features
    return np.hstack((strength, closeness, betweenness, eigenvector, clustering))


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





