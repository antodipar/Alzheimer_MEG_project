
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
Y = data[:,1] - 1

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
densities = [70, 80, 90, 100]
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

X = X[:,:,-1]


# *******************************************
# *******************************************
# *******************************************
# Select model and estimate classification performance using cross-validation
# *******************************************
# *******************************************
# *******************************************
nsamples, nfeats = X.shape

# initialize variables
nfolds = 10 # 10-fold cross-validation
NUM_TRIALS = 1 # repeated 10-fold cross-validation

# --- select the classifier
from MKLpy.algorithms import EasyMKL, AverageMKL
from sklearn.svm import SVC
clf = SVC(C = 1e-2, class_weight='balanced', random_state = 1, kernel= 'linear')


# variables to store the outcome
test_AUC = np.zeros((NUM_TRIALS, nfolds))
training_AUC = np.zeros((NUM_TRIALS, nfolds)) # to have an idea about overfitting
# other metrics to evaluate the classification performance
test_ACC = np.zeros((NUM_TRIALS, nfolds))
test_F1 = np.zeros((NUM_TRIALS, nfolds))

best_features=np.zeros(nfeats)
Nfeats = np.zeros(nfolds)
Y_scores = np.zeros(nsamples)
# start the cross-validation procedure
for itrial in np.arange(NUM_TRIALS):

    print "Trial {} (out of {})".format(itrial+1,NUM_TRIALS)

    # split the dataset into nfols
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = itrial)

    icv = 0
    for indxtrain, indxtest in skf.split(X, Y):

        print "Iteration {} (out of {})".format(icv + 1, nfolds)

        import feature_selection as fs
        reload(fs)

        features = fs.rfs(X[indxtrain, :], Y[indxtrain], NUM_TRIALS=1)
        Nfeats[icv] = len(features)
        best_features[features] +=1

        # --- train the model using the training folds
        clf.fit(X[indxtrain[:,np.newaxis], features], Y[indxtrain])

        # --- test the model in the remaining fold
        from sklearn.metrics import roc_auc_score, roc_curve, f1_score
        y_scores = clf.decision_function(X[indxtest[:,np.newaxis], features])
        y_predict = clf.predict(X[indxtest[:,np.newaxis], features])
        y_true = Y[indxtest]
        test_auc = roc_auc_score(y_true, y_scores)
        test_AUC[itrial, icv] = test_auc
        test_ACC[itrial, icv] = clf.score(X[indxtest[:,np.newaxis], features], y_true)
        test_F1[itrial, icv] = f1_score(y_true, y_predict)
        Y_scores[indxtest]=y_scores
        # --- test the model on the training samples too
        y_scores = clf.decision_function(X[indxtrain[:,np.newaxis], features])
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


# Extract the most discriminative features
nfeat_opt = int(round(np.mean(Nfeats)))
BESTFEATS = np.reshape(best_features, (5,102))
BESTFEATS = np.sum(BESTFEATS, axis=0)
indx = np.argsort(BESTFEATS)[::-1]
print "Optimal number of features: {} - Best_sensors: {}".format(nfeat_opt, indx[:nfeat_opt]+1)


# ROC curve
from scipy import stats
fpr, tpr, thr = roc_curve(Y, Y_scores)
plt.plot(fpr, tpr, color='b',
         label=r'ROC',
         lw=2)

auc = roc_auc_score(Y, Y_scores)
neg=Y_scores[Y==0]
pos=Y_scores[Y==1]
# AUC null
U, pauc=stats.mannwhitneyu(pos,neg, alternative="greater")
print "AUC = {} (P-value = {})".format(auc,pauc)


# optimal point
from scipy.spatial.distance import cdist

D = cdist(np.array([0, 1.0])[np.newaxis, :], np.concatenate((fpr[:, np.newaxis], tpr[:, np.newaxis]), axis=1),
          'euclidean')

optpoint = np.argmin(D)
thropt = thr[optpoint]
cx = fpr[optpoint]
cy = tpr[optpoint]
sensibility = cy
psens = stats.binom_test(round(sensibility * 78), 78, alternative="greater")
specificity = 1 - cx
pspec = stats.binom_test(round(specificity * 54), 54, alternative="greater")
accuracy = (sensibility * 78 + specificity * 54) / (132)
pacc = stats.binom_test(round(sensibility * 78 + specificity * 54), 132, alternative="greater")
print "Optimal threshold = {}".format(thropt)
print "Optimal point. Accuracy = %0.2f (P-value=%0.2e). Sensibility = %0.2f (P-value=%0.2e). Specificity = %0.2f (P-value=%0.2e)" \
      % (accuracy * 100, pacc, sensibility * 100, psens, specificity * 100, pspec)
plt.scatter(cx, cy, color='r', linewidths=1, alpha=1, zorder=5, marker="o", label='Optimal point')

# customize plot
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('ROC.pdf')
plt.savefig('ROC.tif', dpi=500)
plt.show()

