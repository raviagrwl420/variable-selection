#uses method from https://gist.github.com/agramfort/2071994
import itertools
import numpy as np
import pylab as pl

from sklearn import svm, linear_model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC

from scipy import stats

def transform_pairwise(X, y):
    X_new = []
    y_new = []
    
    y = np.asarray(y)
    
    # The comparison is done only within a group
    # If no group information attach dummy groups
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    
    # Generate combinations
    comb = itertools.combinations(range(X.shape[0]), 2)
    
    # Filter and balance the combinations
    for k, (i, j) in enumerate(comb):
        # Skip if same target or different group
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            continue

        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))

        # Output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = -y_new[-1]
            X_new[-1] = -X_new[-1]

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
    def __init__(self):
        super(RankSVM, self).__init__()
        self.kernel = 'poly'
        self.degree = 2
        self.C = .1
        self.decision_function_shape = 'ovo'
        self.gamma = 'scale'

    def fit(self, X, y, sample_weight=None):
        X_trans, y_trans = transform_pairwise(X, y)
        
        if not np.any(sample_weight):
            super(RankSVM, self).fit(X_trans, y_trans)
        else:
            super(RankSVM, self).fit(X_trans, y_trans, sample_weight=sample_weight)

        return self

    def predict(self, X):
        return super(RankSVM, self).predict(X)

    def rank_ovr(self, X, index):
        num_samples = X.shape[0]
        num_features = X.shape[1]

        rest_indices = np.delete(np.arange(num_samples), index)

        one_sample = X[index].reshape(1, -1)
        rest_samples = X[rest_indices]

        return np.mean(self.predict(one_sample-rest_samples))

    def max_rank(self, X):
        score = np.asarray([])
        for i in range(X.shape[0]):
            score = np.append(score, self.rank_ovr(X, i))

        return np.argmax(score)

r = RankSVM()

X = np.random.randn(100, 20)
y = np.r_[np.ones(50), np.zeros(50)]

r.fit(X, y)

r.max_rank(X)