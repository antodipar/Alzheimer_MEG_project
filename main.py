
# NOTE: I will be using Python 2.7.12

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
n = 5151  # # total number of links: 5151. Total number of features: 5151 x 5 = 25.755

if doEda:

    import eda
    reload(eda)
    eda.edges(X, n)

# Given the previous analysis, remove median and mad variables


# *******************************************
# *******************************************
# *******************************************
# Network reconstruction
# *******************************************
# *******************************************
# *******************************************


import network_analysis as netanalysis
reload(netanalysis)

Xlinks = X[:, 0:n]
nsamples, nfeats = Xlinks.shape
nROIs = 102
mask = np.triu(np.ones((nROIs, nROIs), dtype=bool), k=1)
NETS = np.zeros((nsamples, nROIs, nROIs)) # variable to store subject-specific networks


for isubj in np.arange(nsamples):

    W = netanalysis.reconstruct_net(Xlinks[isubj,:], nROIs)
    NETS[isubj] = W


# Feature extraction based on network measures
densities = [50, 60, 70, 80, 90, 100]
nofmetrics = 5 # degree, closeness, ...
X = np.zeros((nsamples, nROIs*nofmetrics, len(densities)))
Xlinks_den = np.zeros((nsamples, nfeats, len(densities)))

for isubj in np.arange(nsamples):

    print "Subject {} (out of {})".format(isubj + 1, nsamples)

    for iden, den in enumerate(densities):

        W = netanalysis.thresholding(NETS[isubj], den)
        # metrics = netanalysis.compute_metrics(W)
        # X[isubj,:,iden] = metrics
        Xlinks_den[isubj, :, iden] = W[mask]



# # save data
# import pickle
# fout=open('data.txt', 'w')
# pickle.dump([X,Y], fout)
# fout.close()

# read from disk
import pickle
fin=open('data.txt', 'r')
X,Y=pickle.load(fin)
fin.close()


# *******************************************
# *******************************************
# *******************************************
# Select model and estimate classification performance using cross-validation
# *******************************************
# *******************************************
# *******************************************
# Now we have a new set of features (if doNet is True)
nsamples, nfeats, _ = X.shape

# initialize variables
nfolds = 10 # 10-fold cross-validation
NUM_TRIALS = 1 # 10-repeated 10-fold cross-validation

# --- select the classifier
from MKLpy.algorithms import EasyMKL, AverageMKL
from sklearn.svm import SVC
from grakel import GraphKernel

wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 10}, {"name": "subtree_wl"}], normalize=True)
clfsvm = SVC(C = 1e-2, class_weight='balanced', random_state = 1, kernel= 'precomputed')
# clf = EasyMKL(estimator=clfsvm, lam=0.1)
clf = AverageMKL(estimator=clfsvm)


# variables to store the outcome
test_AUC = np.zeros((NUM_TRIALS, nfolds))
training_AUC = np.zeros((NUM_TRIALS, nfolds)) # to have an idea about overfitting
# other metrics to evaluate the classification performance
test_ACC = np.zeros((NUM_TRIALS, nfolds))
test_F1 = np.zeros((NUM_TRIALS, nfolds))


# start the cross-validation procedure
for itrial in np.arange(NUM_TRIALS):

    print "Trial {} (out of {})".format(itrial+1,NUM_TRIALS)

    # split the dataset into nfols
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = itrial)

    icv = 0
    for indxtrain, indxtest in skf.split(X, Y):

        print "Iteration {} (out of {})".format(icv + 1, nfolds)
        KL = list()
        for iden, den in enumerate(densities):

            print "Density {}".format(den)

            import feature_selection as fs
            reload(fs)

            if den != 50 and den != 60 and den != 70 and den != 80:

                features = fs.rfs(X[indxtrain, :, iden], Y[indxtrain], NUM_TRIALS=1)
               # generate kernel
                from sklearn.metrics.pairwise import linear_kernel
                KL.append(linear_kernel(X[:, features, iden]))

        Ktrain = [K[indxtrain][:, indxtrain] for K in KL]
        Ktest = [K[indxtest][:, indxtrain] for K in KL]

        # # *******************************
        # # Graph Kernel
        # # *******************************
        # import networkx as nx
        #
        # for iden, den in enumerate(densities):
        #
        #     if den!= 50 and den != 60 and den!= 100:
        #
        #         ADJ = netanalysis.maxcc(Xlinks_den[indxtrain,:, iden], Y[indxtrain], th = 1)
        #
        #         # training samples
        #         training_NETS = list()
        #         for indx in indxtrain:
        #             net = netanalysis.reconstruct_net(Xlinks_den[indx,:,iden],nROIs)
        #             W = nx.Graph(net * ADJ)
        #             st = W.degree()
        #             edge = {key: '{}'.format(value) for key, value in st}
        #             training_NETS.append([net, edge])
        #
        #         # test samples
        #         test_NETS = list()
        #         for indx in indxtest:
        #             net = netanalysis.reconstruct_net(Xlinks_den[indx,:,iden],nROIs)
        #             W = nx.Graph(net * ADJ)
        #             st = W.degree()
        #             edge = {key: '{}'.format(value) for key, value in st}
        #             test_NETS.append([net, edge])
        #
        #         training_K = wl_kernel.fit_transform(training_NETS)
        #         test_K = wl_kernel.transform(test_NETS)
        #
        #         Ktrain.append(training_K)
        #         Ktest.append(test_K)

        # *******************************
        # *******************************

        # --- train the model using the training folds
        clf.fit(Ktrain, Y[indxtrain])

        # --- test the model in the remaining fold and compute the ROC curve
        from sklearn.metrics import roc_auc_score, f1_score
        y_scores = clf.decision_function(Ktest)
        y_predict = clf.predict(Ktest)
        y_true = Y[indxtest]
        test_auc = roc_auc_score(y_true, y_scores)
        test_AUC[itrial, icv] = test_auc
        test_ACC[itrial, icv] = clf.score(Ktest, y_true)
        test_F1[itrial, icv] = f1_score(y_true, y_predict)

        # --- test the model on the training samples too
        y_scores = clf.decision_function(Ktrain)
        y_true = Y[indxtrain]
        training_auc = roc_auc_score(y_true, y_scores)
        training_AUC[itrial, icv] = training_auc

        print "Test AUC: {} - Training AUC: {}\n".format(test_auc, training_auc)

        icv += 1

# # *******************************************
# # *******************************************
# # *******************************************
# # Outcome
# # *******************************************
# # *******************************************
# # *******************************************
mean_test_auc = np.mean(test_AUC)
mean_training_auc = np.mean(training_AUC)


print "\n\nRESULT - Test AUC: {} - Training AUC: {}\n\n".format(mean_test_auc, mean_training_auc)





