

import numpy as np
import matplotlib.pyplot as plt


def select_features(X,Y, prct=20, nfolds=10):


    nsamples, nfeats = X.shape

    # select classifier
    from sklearn.svm import SVC
    clf = SVC(C=1e-2, random_state=1, class_weight='balanced', kernel='linear')

    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(C=1e-2, penalty='l2', random_state=1, class_weight='balanced')


    # initialize variables
    nfeats_limit = int(round(prct * 1.0 / 100 * nfeats))
    importance_scores = np.zeros((nfolds, nfeats)) # store the importance of each feature across repetitions
    test_AUC = np.zeros((nfolds, nfeats))
    training_AUC = np.zeros((nfolds, nfeats)) # to have an idea about overfitting

    # split the dataset into nfolds
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = 0)

    icv = 0
    for indxtrain, indxtest in skf.split(X, Y):


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
        importance_scores[icv, :] = importance


        for i in range(nfeats_limit): # this for loop implements a recursive feature selection procedure


            # --- train the model using the training folds
            clf.fit(training_samples[:, ranking[0:i + 1]], training_labels)

            # --- test the model in the remaining fold and compute the ROC curve
            from sklearn.metrics import roc_auc_score
            y_scores = clf.decision_function(test_samples[:, ranking[0:i + 1]])
            y_true = test_labels
            test_auc = roc_auc_score(y_true, y_scores)
            test_AUC[icv, i] = test_auc

            # --- test the model on the training samples too
            y_scores = clf.decision_function(training_samples[:, ranking[0:i + 1]])
            y_true = training_labels
            training_auc = roc_auc_score(y_true, y_scores)
            training_AUC[icv, i] = training_auc

            # print "Test AUC: {} - Training AUC: {}".format(test_auc, training_auc)

        icv += 1



    # compute the average importance scores
    mean_fi = np.mean(importance_scores,0)
    final_ranking = np.argsort(mean_fi)
    final_ranking = final_ranking[::-1]

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
    print "Max Auc %g (sd: %g) with %u features\n" % (mean_test_auc[max_indx], std_test_auc[max_indx], max_indx + 1)

    # plt.figure()
    #
    # plt.plot(np.arange(1, nfeats + 1), mean_test_auc, color='b', lw=2, label="Mean Test AUC")
    # plt.fill_between(np.arange(1,nfeats+1), test_auc_lower, test_auc_upper, color='grey', alpha=.2, label=r'$\pm$ 1 SD.')
    #
    # plt.plot(np.arange(1, nfeats + 1), mean_training_auc, color='g', lw=2, label="Mean Training AUC")
    # plt.fill_between(np.arange(1,nfeats+1), training_auc_lower, training_auc_upper, color='red', alpha=.2, label=r'$\pm$ 1 SD.')
    #
    # plt.scatter(max_indx+1, mean_test_auc[max_indx], color='k', marker="D", linewidths=1, alpha=1, zorder=5, label="Max AUC")
    # plt.xlabel('Number of features')
    # plt.ylabel('AUC')
    # plt.legend(loc="lower right")
    # plt.grid(True)
    # plt.xlim([-2, nfeats+2])
    # plt.ylim([0, 1.1])
    # plt.show()


    return final_ranking[:max_indx+1]
