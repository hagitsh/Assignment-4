import numpy as np
from mrmr import mrmr_classif
from sklearn.svm import SVR
from ReliefF import ReliefF
from sklearn.feature_selection import SelectFdr, f_classif, RFE, SelectKBest


def reliefF(X, y):
    fs = ReliefF(n_neighbors=48)
    fs.fit(X.values, y.values)
    return fs.top_features

def rfe(X, y):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator)
    selector = selector.fit(X, y)
    return list(np.where(selector.support_)[0])


def Fdr_f_classif(X, y):
    fs = SelectFdr(f_classif, alpha=0.1)
    fs.fit(X, y)
    return fs.get_support(True)


def mrmr(X, y):
    class_names =  mrmr_classif(X=X, y=y, K=100)
    return list(np.where([x in class_names for x in X.columns])[0])


def k1_best(X, y):
    fs = SelectKBest(f_classif, k=1000)
    fs.fit(X, y)
    return fs.get_support(True)