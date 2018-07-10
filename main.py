
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
Y = data[:,1]
Y=Y-1
# Y = data[:,1] > 1

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

X = X[:, 0:n]
nsamples, nfeats = X.shape
nROIs = 102
NETS = np.zeros((nsamples, nROIs, nROIs)) # variable to store subject-specific networks



for isubj in np.arange(nsamples):

    W = netanalysis.reconstruct_net(X[isubj,:], nROIs)
    NETS[isubj] = W


# Feature extraction based on network measures
densities = [50, 60, 70, 80, 90, 100]
nofmetrics = 5 # degree, closeness, ...
newX = np.zeros((nsamples, nROIs*nofmetrics, len(densities)))

# for isubj in np.arange(nsamples):
#
#     print "Subject {} (out of {})".format(isubj + 1, nsamples)
#
#     for iden, den in enumerate(densities):
#
#         W = netanalysis.thresholding(NETS[isubj], den)
#         metrics = netanalysis.compute_metrics(W)
#         newX[isubj,:,iden] = metrics
#
# X = newX
#
# # save data
# import pickle
# fout=open('data.txt', 'w')
# pickle.dump([X,Y], fout)
# fout.close()

# read from disk
import pickle
fin=open('data.txt', 'r')
X,_=pickle.load(fin)
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

# --- select the classifier
from MKLpy.algorithms import AverageMKL
from sklearn.svm import SVC
clfsvm = SVC(C = 1e-2, class_weight='balanced', random_state = 1, kernel= 'precomputed')
clf = AverageMKL(estimator=clfsvm)


# variables to store the outcome
test_SCORES = np.zeros((nsamples))
test_ACC = np.zeros((nsamples))
# to have an idea about overfitting
training_ACC = np.zeros((nsamples))


# start the cross-validation procedure
from sklearn.model_selection import LeaveOneOut
loo =  LeaveOneOut()

icv = 0
for indxtrain, indxtest in loo.split(X[:,:,0]):

    print "Subject {} (out of {})".format(icv + 1, nsamples)
    KL = list()
    for iden, den in enumerate(densities):

        print "Density {}".format(den)

        import feature_selection as fs
        reload(fs)
        if den != 50 and den != 60:
            features = fs.rfs(X[indxtrain, :, iden], Y[indxtrain])
           # generate kernel
            from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
            KL.append(linear_kernel(X[:, features, iden]))

    Ktrain = [K[indxtrain][:, indxtrain] for K in KL]
    Ktest = [K[indxtest][:, indxtrain] for K in KL]

    # --- train the model using the training folds
    clf.fit(Ktrain, Y[indxtrain])

    # --- test the model in the remaining fold and compute the ROC curve
    from sklearn.metrics import roc_auc_score, f1_score
    y_scores = clf.decision_function(Ktest)
    y_true = Y[indxtest]
    print "True: {} - Score: {}".format(y_true, y_scores)
    test_SCORES[icv] = y_scores
    test_ACC[icv] = clf.score(Ktest, y_true)

    # --- test the model on the training samples too
    y_true = Y[indxtrain]
    training_ACC[icv] = clf.score(Ktrain, y_true)

    print "Test ACC: {} - Training ACC: {}\n".format(test_ACC[icv], training_ACC[icv])

    icv += 1

# # *******************************************
# # *******************************************
# # *******************************************
# # Outcome
# # *******************************************
# # *******************************************
# # *******************************************

mean_test_acc = np.mean(test_ACC)
mean_training_acc = np.mean(training_ACC)


print "\n\nRESULT - Test ACC: {} - Training ACC: {}\n\n".format(mean_test_acc, mean_training_acc)
print "AUC: ", roc_auc_score(Y, test_SCORES)




