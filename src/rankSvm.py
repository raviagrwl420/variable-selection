from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from scipy import stats
import pylab as pl
from sklearn import svm, linear_model
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

def transform_pairwise(X,y):
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()

class RankSVM(svm.SVC):
    
    def fit(self,X,y):
        X_trans, y_trans = transform_pairwise(X, y)
        self.kernel='poly'
        self.degree=2
        self.C=.1
        super(RankSVM, self).fit(X_trans, y_trans)
        return self
    
    def predict(self, X):
        if hasattr(self, 'coef_'):
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)
