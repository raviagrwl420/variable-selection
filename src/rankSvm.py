from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC




class svm(object):
    def rank(X,y):
        kf = KFold(n_splits=len(X[1]))
        for train, test in kf.split(X):
            print("%s %s" % (train, test))
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf = SVC(gamma='auto')
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)