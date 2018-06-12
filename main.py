
# NOTE: I will be using Python 2.7.6

# import basic modules
import matplotlib.pyplot as plt
plt.ioff() # turn the interactive mode off
import numpy as np
import pandas as pd
import math

# load data
Data = pd.read_csv('meg_mci.csv')
# ---
names = Data.keys()
data = Data.values
X = data[:,2:]
Y = data[:,1] > 1


# *******************************************
# *******************************************
# *******************************************
# exploratory data analysis
# *******************************************
# *******************************************
# *******************************************

flag = False
n = 5151  # total number of features: 5151 x 5 = 25.755
feat_mean = X[:, 0:n]
feat_std = X[:, n:2*n]
feat_median = X[:, 2*n:3*n]
feat_mad = X[:, 3*n:4*n]
feat_cov = X[:, 4*n:5*n]

if flag:


    # compute pair-wise correlation to uncover linear dependencies

    # Mean & std
    corr_mean_std = np.zeros((n))
    for i in np.arange(n):
        corr_mean_std[i] = np.corrcoef(feat_mean[:,i], feat_std[:,i], rowvar = False)[0,1]

    # Mean & median
    corr_mean_median = np.zeros((n))
    for i in np.arange(n):
        corr_mean_median[i] = np.corrcoef(feat_mean[:,i], feat_median[:,i], rowvar = False)[0,1]

    # Mean & mad
    corr_mean_mad = np.zeros((n))
    for i in np.arange(n):
        corr_mean_mad[i] = np.corrcoef(feat_mean[:,i], feat_mad[:,i], rowvar = False)[0,1]

    # Mean & cov
    corr_mean_cov = np.zeros((n))
    for i in np.arange(n):
        corr_mean_cov[i] = np.corrcoef(feat_mean[:,i], feat_cov[:,i], rowvar = False)[0,1]

    # Std & median
    corr_std_median = np.zeros((n))
    for i in np.arange(n):
        corr_std_median[i] = np.corrcoef(feat_std[:,i], feat_median[:,i], rowvar = False)[0,1]

    # Std & mad
    corr_std_mad = np.zeros((n))
    for i in np.arange(n):
        corr_std_mad[i] = np.corrcoef(feat_std[:,i], feat_mad[:,i], rowvar = False)[0,1]

    # Std & cov
    corr_std_cov = np.zeros((n))
    for i in np.arange(n):
        corr_std_cov[i] = np.corrcoef(feat_std[:,i], feat_cov[:,i], rowvar = False)[0,1]

    # Median & mad
    corr_median_mad = np.zeros((n))
    for i in np.arange(n):
        corr_median_mad[i] = np.corrcoef(feat_median[:,i], feat_mad[:,i], rowvar = False)[0,1]

    # Median & cov
    corr_median_cov = np.zeros((n))
    for i in np.arange(n):
        corr_median_cov[i] = np.corrcoef(feat_median[:,i], feat_cov[:,i], rowvar = False)[0,1]

    # Mad & cov
    corr_mad_cov = np.zeros((n))
    for i in np.arange(n):
        corr_mad_cov[i] = np.corrcoef(feat_mad[:,i], feat_cov[:,i], rowvar = False)[0,1]

    # Visualize data

    # Mean & std
    nbins = 20

    plt.figure()
    plt.subplot(4,4,1)
    plt.hist(corr_mean_std, bins = nbins)
    plt.xlabel('Corr mean-std')
    plt.ylabel('Counts')
    plt.show()

    # Mean & median
    plt.subplot(4,4,2)
    plt.hist(corr_mean_median, bins = nbins)
    # plt.xlabel('Corr mean-median')
    # plt.ylabel('Counts')
    plt.show()

    # Mean & mad
    plt.subplot(4,4,3)
    plt.hist(corr_mean_mad, bins = nbins)
    # plt.xlabel('Corr mean-mad')
    # plt.ylabel('Counts')
    plt.show()

    # Mean & cov
    plt.subplot(4,4,4)
    plt.hist(corr_mean_cov, bins = nbins)
    # plt.xlabel('Corr mean-cov')
    # plt.ylabel('Counts')
    plt.show()

    # Std & median
    plt.subplot(4,4,4+2)
    plt.hist(corr_std_median, bins = nbins)
    plt.xlabel('Corr std-median')
    plt.ylabel('Counts')
    plt.show()

    # Std & mad
    plt.subplot(4,4,4+3)
    plt.hist(corr_std_mad, bins = nbins)
    # plt.xlabel('Corr std-mad')
    # plt.ylabel('Counts')
    plt.show()

    # Std & cov
    plt.subplot(4,4,4+4)
    plt.hist(corr_std_cov, bins = nbins)
    # plt.xlabel('Corr std-cov')
    # plt.ylabel('Counts')
    plt.show()

    # Median & mad
    plt.subplot(4,4,8+3)
    plt.hist(corr_median_mad, bins = nbins)
    plt.xlabel('Corr median-mad')
    plt.ylabel('Counts')
    plt.show()

    # Median & cov
    plt.subplot(4,4,8+4)
    plt.hist(corr_median_cov, bins = nbins)
    # plt.xlabel('Corr median-cov')
    # plt.ylabel('Counts')
    plt.show()

    # Mad & cov
    plt.subplot(4,4,12+4)
    plt.hist(corr_mad_cov, bins = nbins)
    plt.xlabel('Corr mad-cov')
    plt.ylabel('Counts')
    plt.show()

# Given the previous analysis, remove median and mad variables
X = np.hstack((X[:,0:n*2], X[:,4*n:5*n]))

# preprocess data? Let's have a look at the raw values of mean, std and cov

plt.figure()
plt.boxplot(np.hstack((feat_mean.flatten()[:,np.newaxis], feat_std.flatten()[:,np.newaxis], feat_cov.flatten()[:,np.newaxis])))
plt.show()

# *******************************************
# *******************************************
# *******************************************
# Select model and estimate classification performance using cross-validation
# *******************************************
# *******************************************
# *******************************************

# initialize variables
nfolds = 2 # 10-fold cross-validation
NUM_TRIALS = 1 # 10-repeated 10-fold cross-validation
nsamples, nfeats = X.shape
prctile=90
nfeats_limit = int(math.ceil((100-prctile*1.0)/100*nfeats)) # max number of features to be used

importance_scores = np.zeros((NUM_TRIALS * nfolds, nfeats)) # store the importance of each feature across repetitions

# --- select the classifier
from sklearn.svm import SVC, LinearSVC
clf = LinearSVC(C = 1, random_state = 1, class_weight = 'balanced')  # gaussian kernel with C=1 and sigma=1/num_features

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, random_state=1, class_weight = 'balanced')
# ---

# variables to generate ROC curves
FPR = dict()
TPR = dict()
test_AUC = np.zeros((NUM_TRIALS*nfolds, nfeats))

# start the cross-validation procedure
for itrial in np.arange(NUM_TRIALS):

    print "Trial {} (out of {})".format(itrial+1,NUM_TRIALS)

    # split the dataset into 10 folds (9 for training and 1 for testing)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = itrial)

    icv = 0
    for fold, (indxtrain, indxtest) in enumerate(skf.split(X, Y)):

        # calculate feature ranking
        import feature_importance as fi
        importance = fi.ttest(X[indxtrain,:], Y[indxtrain])
        thr = np.percentile(importance, prctile)
        importance[np.where(importance < thr)[0]] = 0

        # ---
        importance_scores[itrial * nfolds + icv, :] = importance
        ranking = np.argsort(importance)
        ranking = ranking[::-1]

        for i in range(nfeats_limit): # this for loop implements a recursive feature selection procedure

            if i % 100 == 0:
                print "Iteration {} (out of {}). Feature {} (out of {} features)".format(fold+1, nfolds, i+1, nfeats_limit)
            # --- train the model using the training folds
            # from sklearn import preprocessing
            # scaler=preprocessing.StandardScaler().fit(X[indxtrain[:,np.newaxis], ranking[0:i + 1]])
            clf.fit((X[indxtrain[:,np.newaxis], ranking[0:i + 1]]), Y[indxtrain])
            # --- test the model in the remaining fold and compute the ROC curve
            from sklearn.metrics import roc_auc_score, roc_curve
            # y_scores = clf.predict_proba( (X[indxtest[:,np.newaxis], ranking[0:i + 1]]) )[:, 1]
            y_scores = clf.decision_function( (X[indxtest[:,np.newaxis], ranking[0:i + 1]]) )
            y_true = Y[indxtest]
            fpr, tpr, thr = roc_curve(y_true, y_scores)
            FPR[itrial * nfolds + icv, i] = list(fpr)
            TPR[itrial * nfolds + icv, i] = list(tpr)
            test_AUC[itrial * nfolds + icv, i] = roc_auc_score(y_true, y_scores)

        icv += 1


# *******************************************
# *******************************************
# *******************************************
# Visualize the outcome
# *******************************************
# *******************************************
# *******************************************

# # compute the average importance scores
# mean_fi = np.mean(importance_scores,0)
# final_ranking = np.argsort(mean_fi)
# final_ranking = final_ranking[::-1]

# select features and plot test_AUC
mean_test_auc = np.mean(test_AUC, 0)
std_test_auc = np.std(test_AUC, 0)

test_auc_upper = mean_test_auc + std_test_auc
test_auc_lower = mean_test_auc - std_test_auc

max_indx = np.argmax(mean_test_auc) # 'max_indx + 1' is the optimal number of features to be used for diagnosis
print "\nMax Auc %g (sd: %g) with %u features" % (mean_test_auc[max_indx], std_test_auc[max_indx], max_indx + 1)
plt.plot(np.arange(1, nfeats + 1), mean_test_auc, color='b', lw=2, label="Mean Performance Profile")
plt.fill_between(np.arange(1,nfeats+1), test_auc_lower, test_auc_upper, color='grey', alpha=.2, label=r'$\pm$ 1 SD.')
plt.scatter(max_indx+1, mean_test_auc[max_indx], color='k', marker="D", linewidths=1, alpha=1, zorder=5, label="Max AUC")
plt.xlabel('Number of features')
plt.ylabel('AUC')
plt.legend(loc="lower right")
plt.grid(True)
plt.xlim([-2, nfeats+2])
plt.ylim([0, 1])
plt.show()