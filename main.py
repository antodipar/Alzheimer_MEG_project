
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

for isubj in np.arange(nsamples):

    W = netanalysis.reconstruct_net(X[isubj,:], nROIs)
    NETS[isubj] = W



# Feature extraction based on network measures
densities = [50, 60, 70, 80, 90]
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
prct = 20 # percentage of features to be used
nfeats_limit = int(round(prct*1.0/100*nfeats))
scaling = False # feature scaling?

importance_scores = np.zeros((NUM_TRIALS * nfolds, nfeats)) # store the importance of each feature across repetitions

# --- select the classifier
from MKLpy.algorithms import EasyMKL
from sklearn.svm import SVC
clfsvm = SVC(C = 1, class_weight='balanced', random_state = 1, kernel= 'precomputed')
clf = EasyMKL(estimator=clfsvm, lam=0)

# ---

# variables to generate ROC curves
FPR = dict()
TPR = dict()
test_AUC = np.zeros((NUM_TRIALS*nfolds, nfeats))
training_AUC = np.zeros((NUM_TRIALS*nfolds, nfeats)) # to have an idea about overfitting
# other metrics to evaluate the classification performance
test_ACC = np.zeros((NUM_TRIALS*nfolds, nfeats))
test_F1 = np.zeros((NUM_TRIALS*nfolds, nfeats))

# start the cross-validation procedure
for itrial in np.arange(NUM_TRIALS):

    print "Trial {} (out of {})".format(itrial+1,NUM_TRIALS)

    # split the dataset into nfols
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = itrial)

    icv = 0
    for indxtrain, indxtest in skf.split(X, Y):

        print "Iteration {} (out of {})".format(icv + 1, nfolds)


        # take training and test samples first
        training_samples = X[indxtrain, :]
        training_labels = Y[indxtrain]
        test_samples = X[indxtest, :]
        test_labels = Y[indxtest]

        # compute feature ranking
        import feature_importance as fi
        reload(fi)

        importance = fi.wilcoxon(training_samples, training_labels)
        ranking = np.argsort(importance)
        ranking = ranking[::-1]
        importance_scores[itrial * nfolds + icv, :] = importance


        # preprocess the data
        if scaling:
            from sklearn import preprocessing
            scaler=preprocessing.StandardScaler().fit(training_samples)
            training_samples = scaler.transform(training_samples)
            test_samples = scaler.transform(test_samples)


        for i in range(nfeats_limit): # this for loop implements a recursive feature selection procedure

            if i % 100 == 0:
                print "Features {} (out of {})".format(i+1, nfeats_limit)

            from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
            KL = [linear_kernel(X[:, ranking[0:i + 1]]), rbf_kernel(X[:, ranking[0:i + 1]])]
            Ktrain = [K[indxtrain][:, indxtrain] for K in KL]
            Ktest = [K[indxtest][:, indxtrain] for K in KL]

            # --- train the model using the training folds
            clf.fit(Ktrain, training_labels)


            # --- test the model in the remaining fold and compute the ROC curve
            from sklearn.metrics import roc_auc_score, roc_curve, f1_score
            y_scores = clf.decision_function( Ktest)
            # y_scores = clf.predict_proba(test_samples[:, ranking[0:i + 1]])[:,1]
            y_predict = clf.predict(Ktest)
            y_true = test_labels
            fpr, tpr, thr = roc_curve(y_true, y_scores)
            FPR[itrial * nfolds + icv, i] = list(fpr)
            TPR[itrial * nfolds + icv, i] = list(tpr)
            test_auc = roc_auc_score(y_true, y_scores)
            test_AUC[itrial * nfolds + icv, i] = test_auc
            test_ACC[itrial * nfolds + icv, i] = clf.score(Ktest, y_true)
            test_F1[itrial * nfolds + icv, i] = f1_score(y_true, y_predict)

            # --- test the model on the training samples too
            y_scores = clf.decision_function(Ktrain)
            # y_scores = clf.predict_proba(training_samples[:, ranking[0:i + 1]])[:,1]
            y_true = training_labels
            training_auc = roc_auc_score(y_true, y_scores)
            training_AUC[itrial * nfolds + icv, i] = training_auc

            print "Test AUC: {} - Training AUC: {}".format(test_auc, training_auc)

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

# select features and plot test_AUC and training_AUC
mean_test_auc = np.mean(test_AUC, 0)
std_test_auc = np.std(test_AUC, 0)
test_auc_upper = mean_test_auc + std_test_auc
test_auc_lower = mean_test_auc - std_test_auc

mean_training_auc = np.mean(training_AUC, 0)
std_training_auc = np.std(training_AUC, 0)
training_auc_upper = mean_training_auc + std_training_auc
training_auc_lower = mean_training_auc - std_training_auc

max_indx = np.argmax(mean_test_auc) # 'max_indx + 1' is the optimal number of features to be used for diagnosis
print "\nMax Auc %g (sd: %g) with %u features" % (mean_test_auc[max_indx], std_test_auc[max_indx], max_indx + 1)
plt.figure()

plt.plot(np.arange(1, nfeats + 1), mean_test_auc, color='b', lw=2, label="Mean Test AUC")
plt.fill_between(np.arange(1,nfeats+1), test_auc_lower, test_auc_upper, color='grey', alpha=.2, label=r'$\pm$ 1 SD.')

plt.plot(np.arange(1, nfeats + 1), mean_training_auc, color='g', lw=2, label="Mean Training AUC")
plt.fill_between(np.arange(1,nfeats+1), training_auc_lower, training_auc_upper, color='red', alpha=.2, label=r'$\pm$ 1 SD.')

plt.scatter(max_indx+1, mean_test_auc[max_indx], color='k', marker="D", linewidths=1, alpha=1, zorder=5, label="Max AUC")
plt.xlabel('Number of features')
plt.ylabel('AUC')
plt.legend(loc="lower right")
plt.grid(True)
plt.xlim([-2, nfeats+2])
plt.ylim([0, 1.1])
plt.show()

# visualiza auc, acc and f1 only in the test dataset
mean_test_acc = np.mean(test_ACC, 0)
mean_test_f1 = np.mean(test_F1, 0)

plt.figure()
plt.plot(np.arange(1, nfeats + 1), mean_test_auc, color='b', lw=2, label="Mean Test AUC")
plt.plot(np.arange(1, nfeats + 1), mean_test_acc, color='g', lw=2, label="Mean Test ACC")
plt.plot(np.arange(1, nfeats + 1), mean_test_f1, color='k', lw=2, label="Mean Test F1")

plt.xlabel('Number of features')
plt.ylabel('Performance')
plt.legend(loc="lower right")
plt.grid(True)
plt.xlim([-2, nfeats+2])
plt.ylim([0, 1.1])
plt.show()