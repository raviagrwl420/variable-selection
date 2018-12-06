#uses method from https://gist.github.com/agramfort/2071994
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

#input vector of X and y train and test vectors 
def max_rank(X,y):
    #this function trains the svm model using X[0]
    #it thens uses this model to rank the other vectors in X
    #returns the vector with the highest rank
    result=[]
    cv = StratifiedShuffleSplit(test_size=.5)
    for train, test in cv.split(X[0],y[0]):
        X_train, y_train = X[0][train], y[0][train]
        X_test, y_test = X[0][test], y[0][test]
    Y = np.c_[y[0], np.mod(np.arange(len(y[0])), 2)]  # add query fake id
    rank_svm = RankSVM().fit(X[0][train], Y[train])
    for i,j in zip(X,y):
        cv = StratifiedShuffleSplit(test_size=.5)
        for train, test in cv.split(i,j):
            X_train, y_train = i[train], j[train]
            X_test, y_test = i[test], j[test]
        Y = np.c_[j, np.mod(np.arange(len(j)), 2)]  # add query fake id
        result.append(rank_svm.rank(i[test], j[test]))
    return X[result.index(max(result))]
        
        
    
class RankSVM(svm.SVC):
    def fit(self,X,y,sample_weight=None):
        X_trans, y_trans = transform_pairwise(X, y)
        self.kernel='poly'
        self.degree=2
        self.C=.1
        if not np.any(sample_weight):
            super(RankSVM, self).fit(X_trans, y_trans)
        else:
            super(RankSVM, self).fit(X_trans, y_trans,sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        if hasattr(self, 'coef_'):
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def rank(self, X, y):
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)

