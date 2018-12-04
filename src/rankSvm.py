#from http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from scipy import stats
import pylab as pl
from sklearn import svm, linear_model
import itertools
from sklearn.model_selection import train_test_split

def rank(X,y,blocks):
    cv = StratifiedShuffleSplit(test_size=.5)
    for train, test in cv.split(X,y):
        X_train, y_train, b_train = X[train], y[train], blocks[train]
        X_test, y_test, b_test = X[test], y[test], blocks[test]
    comb = itertools.combinations(range(X_train.shape[0]), 2)
    Xp, yp, diff = [], [], []
    for k, (i, j) in enumerate(comb):
        if y_train[i] == y_train[j] \
            or blocks[train][i] != blocks[train][j]:
            # skip if same target or different group
            continue
        Xp.append(X_train[i] - X_train[j])
        diff.append(y_train[i] - y_train[j])
        yp.append(np.sign(diff[-1]))
        # output balanced classes
        if yp[-1] != (-1) ** k:
            yp[-1] *= -1
            Xp[-1] *= -1
            diff[-1] *= -1
    Xp,yp,diff = map(np.asanyarray, (Xp, yp, diff))
    clf=svm.SVC(kernel='linear',C=.1)
    clf.fit(Xp,yp)
    coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)
    rank={}
    for i in range(2):
        tau, _ = stats.kendalltau(
            np.dot(X_test[b_test == i], coef), y_test[b_test == i])
            rank[i]=tau
    return rank
        
        



