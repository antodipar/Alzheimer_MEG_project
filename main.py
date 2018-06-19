
# NOTE: I will be using Python 2.7.6

# import basic modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
# Exploratory data analysis
# *******************************************
# *******************************************
# *******************************************

doEda = False
n = 5151  # total number of features: 5151 x 5 = 25.755

if doEda:

    import eda
    reload(eda)
    eda.edges(X, n)

# Given the previous analysis, remove median and mad variables
# X = np.hstack((X[:,0:n*2], X[:,4*n:5*n]))
X = X[:,0:n]


# *******************************************
# *******************************************
# *******************************************
# Network reconstruction
# *******************************************
# *******************************************
# *******************************************
import network_analysis as netanalysis
reload(netanalysis)

nsamples, nfeats = X.shape
nROIs = 102
NETS = np.zeros((nsamples, nROIs, nROIs)) # variable to store subject-specific networks
mask = np.triu(np.ones((nROIs, nROIs), dtype=bool), k=1)
for isubj in np.arange(nsamples):

    W = NETS[isubj].copy()
    links = X[isubj,:]
    W[mask] = links
    W = W + W.T
    NETS[isubj] = netanalysis.thresholding(W, [40]).squeeze()
    # NETS[isubj] = W

# *******************************************
# *******************************************
# *******************************************
# Select model and estimate classification performance using cross-validation
# *******************************************
# *******************************************
# *******************************************
nfolds = 10 # 10-fold cross-validation
NUM_TRIALS = 1 # 10-repeated 10-fold cross-validation

# --- select the kernel
from grakel import GraphKernel
wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 10}, {"name": "subtree_wl"}], normalize=True)

# --- now the classifier
from sklearn.svm import SVC
clf = SVC(C = 1e-3, kernel='precomputed', random_state = 1, class_weight = 'balanced')

# ---
# variables to generate ROC curves
FPR = dict()
TPR = dict()
test_AUC = np.zeros((NUM_TRIALS*nfolds))
training_AUC = np.zeros((NUM_TRIALS*nfolds)) # to have an idea about overfitting
# other metrics to evaluate the classification performance
test_ACC = np.zeros((NUM_TRIALS*nfolds))
test_F1 = np.zeros((NUM_TRIALS*nfolds))

# start the cross-validation procedure
for itrial in np.arange(NUM_TRIALS):

    print "Trial {} (out of {})".format(itrial+1,NUM_TRIALS)

    # split the dataset into nfols
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = itrial)

    icv = 0
    for indxtrain, indxtest in skf.split(np.zeros(nsamples), Y):

        # extract cd
        import feature_importance as fi
        ADJ = fi.maxcc(X[indxtrain,:], Y[indxtrain], th=1.5)
        import networkx as nx
        # training samples
        training_labels = Y[indxtrain]
        training_NETS = list()
        for indx in indxtrain:
            W = nx.Graph(NETS[indx]*ADJ)
            st = W.degree()
            edge = {key: '{}'.format(value) for key,value in st}
            training_NETS.append([NETS[indx], edge])

        # test samples
        test_labels = Y[indxtest]
        test_NETS = list()
        for indx in indxtest:
            W = nx.Graph(NETS[indx]*ADJ)
            st = W.degree()
            edge = {key: '{}'.format(value) for key, value in st}
            test_NETS.append([NETS[indx], edge])


        # train the model
        training_K = wl_kernel.fit_transform(training_NETS)
        clf.fit(training_K, training_labels)

        # test the model in the remaining fold and compute the ROC curve
        from sklearn.metrics import roc_auc_score, roc_curve, f1_score
        test_K = wl_kernel.transform(test_NETS)
        y_scores = clf.decision_function(test_K)
        y_predict = clf.predict(test_K)
        y_true = test_labels
        fpr, tpr, thr = roc_curve(y_true, y_scores)
        FPR[itrial * nfolds + icv] = list(fpr)
        TPR[itrial * nfolds + icv] = list(tpr)
        test_AUC[itrial * nfolds + icv] = roc_auc_score(y_true, y_scores)
        test_ACC[itrial * nfolds + icv] = clf.score(test_K, y_true)
        test_F1[itrial * nfolds + icv] = f1_score(y_true, y_predict)

        # --- test the model on the training samples too
        y_scores = clf.decision_function(training_K)
        y_true = training_labels
        training_AUC[itrial * nfolds + icv] = roc_auc_score(y_true, y_scores)


        icv += 1


# *******************************************
# *******************************************
# *******************************************
# Outcome
# *******************************************
# *******************************************
# *******************************************

print "Test dataset"
print "AUC: {} ({})".format(test_AUC.mean(), test_AUC.std())
print "ACC: {} ({})".format(test_ACC.mean(), test_ACC.std())
print "F1: {} ({})".format(test_F1.mean(), test_F1.std())

print "Training dataset"
print "AUC: {} ({})".format(training_AUC.mean(), training_AUC.std())

