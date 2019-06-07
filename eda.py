
import numpy as np
import matplotlib.pyplot as plt


def edges(X,n):


    feat_mean = X[0:, 0:n]
    feat_std = X[:, n:2 * n]
    feat_median = X[:, 2 * n:3 * n]
    feat_mad = X[:, 3 * n:4 * n]
    feat_cov = X[:, 4 * n:5 * n]


    # compute pair-wise correlation to uncover linear dependencies

    # Mean & std
    corr_mean_std = np.zeros((n))
    for i in np.arange(n):
        corr_mean_std[i] = np.corrcoef(feat_mean[:, i], feat_std[:, i], rowvar=False)[0, 1]

    # Mean & median
    corr_mean_median = np.zeros((n))
    for i in np.arange(n):
        corr_mean_median[i] = np.corrcoef(feat_mean[:, i], feat_median[:, i], rowvar=False)[0, 1]

    # Mean & mad
    corr_mean_mad = np.zeros((n))
    for i in np.arange(n):
        corr_mean_mad[i] = np.corrcoef(feat_mean[:, i], feat_mad[:, i], rowvar=False)[0, 1]

    # Mean & cov
    corr_mean_cov = np.zeros((n))
    for i in np.arange(n):
        corr_mean_cov[i] = np.corrcoef(feat_mean[:, i], feat_cov[:, i], rowvar=False)[0, 1]

    # Std & median
    corr_std_median = np.zeros((n))
    for i in np.arange(n):
        corr_std_median[i] = np.corrcoef(feat_std[:, i], feat_median[:, i], rowvar=False)[0, 1]

    # Std & mad
    corr_std_mad = np.zeros((n))
    for i in np.arange(n):
        corr_std_mad[i] = np.corrcoef(feat_std[:, i], feat_mad[:, i], rowvar=False)[0, 1]

    # Std & cov
    corr_std_cov = np.zeros((n))
    for i in np.arange(n):
        corr_std_cov[i] = np.corrcoef(feat_std[:, i], feat_cov[:, i], rowvar=False)[0, 1]

    # Median & mad
    corr_median_mad = np.zeros((n))
    for i in np.arange(n):
        corr_median_mad[i] = np.corrcoef(feat_median[:, i], feat_mad[:, i], rowvar=False)[0, 1]

    # Median & cov
    corr_median_cov = np.zeros((n))
    for i in np.arange(n):
        corr_median_cov[i] = np.corrcoef(feat_median[:, i], feat_cov[:, i], rowvar=False)[0, 1]

    # Mad & cov
    corr_mad_cov = np.zeros((n))
    for i in np.arange(n):
        corr_mad_cov[i] = np.corrcoef(feat_mad[:, i], feat_cov[:, i], rowvar=False)[0, 1]

    # Visualize data

    # Mean & std
    nbins = 20

    plt.figure()
    plt.subplot(4, 4, 1)
    plt.hist(corr_mean_std, bins=nbins)
    plt.xlabel('Corr mean-std')
    plt.ylabel('Counts')
    plt.show()

    # Mean & median
    plt.subplot(4, 4, 2)
    plt.hist(corr_mean_median, bins=nbins)
    # plt.xlabel('Corr mean-median')
    # plt.ylabel('Counts')
    plt.show()

    # Mean & mad
    plt.subplot(4, 4, 3)
    plt.hist(corr_mean_mad, bins=nbins)
    # plt.xlabel('Corr mean-mad')
    # plt.ylabel('Counts')
    plt.show()

    # Mean & cov
    plt.subplot(4, 4, 4)
    plt.hist(corr_mean_cov, bins=nbins)
    # plt.xlabel('Corr mean-cov')
    # plt.ylabel('Counts')
    plt.show()

    # Std & median
    plt.subplot(4, 4, 4 + 2)
    plt.hist(corr_std_median, bins=nbins)
    plt.xlabel('Corr std-median')
    plt.ylabel('Counts')
    plt.show()

    # Std & mad
    plt.subplot(4, 4, 4 + 3)
    plt.hist(corr_std_mad, bins=nbins)
    # plt.xlabel('Corr std-mad')
    # plt.ylabel('Counts')
    plt.show()

    # Std & cov
    plt.subplot(4, 4, 4 + 4)
    plt.hist(corr_std_cov, bins=nbins)
    # plt.xlabel('Corr std-cov')
    # plt.ylabel('Counts')
    plt.show()

    # Median & mad
    plt.subplot(4, 4, 8 + 3)
    plt.hist(corr_median_mad, bins=nbins)
    plt.xlabel('Corr median-mad')
    plt.ylabel('Counts')
    plt.show()

    # Median & cov
    plt.subplot(4, 4, 8 + 4)
    plt.hist(corr_median_cov, bins=nbins)
    # plt.xlabel('Corr median-cov')
    # plt.ylabel('Counts')
    plt.show()

    # Mad & cov
    plt.subplot(4, 4, 12 + 4)
    plt.hist(corr_mad_cov, bins=nbins)
    plt.xlabel('Corr mad-cov')
    plt.ylabel('Counts')
    plt.show()

    # Let's have a look at the raw values of mean, std and cov

    plt.figure()
    plt.boxplot(np.hstack(
        (feat_mean.flatten()[:, np.newaxis], feat_std.flatten()[:, np.newaxis], feat_cov.flatten()[:, np.newaxis])))
    plt.show()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.hist(feat_mean.flatten(), bins=nbins)
    plt.subplot(1, 3, 2)
    plt.hist(feat_std.flatten(), bins=nbins)
    plt.subplot(1, 3, 3)
    plt.hist(feat_cov.flatten(), bins=nbins)
    plt.suptitle('Histogram of raw values')
    plt.show()

    # Now represent pair-wise correlation individually within each type of feature
    R_mean = np.corrcoef(feat_mean, rowvar=False)
    R_std = np.corrcoef(feat_std, rowvar=False)
    R_cov = np.corrcoef(feat_cov, rowvar=False)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.hist(R_mean[mask], bins=nbins)
    plt.subplot(1, 3, 2)
    plt.hist(R_std[mask], bins=nbins)
    plt.subplot(1, 3, 3)
    plt.hist(R_cov[mask], bins=nbins)
    plt.suptitle('Histogram of pair-wise correlation values')
    plt.show()