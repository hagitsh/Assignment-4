import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

from mmcc import MMCCModel
import xgboost
from ref_functions import reliefF, rfe, Fdr_f_classif, mrmr, k1_best
from select_and_test import run_select_and_test

filename = '../bioconductor/ayeastCC.csv'
classname = 'Class'
#filename = '../bioconductor/breastCancerVDX.csv'
#classname = 'oestrogenreceptorsClass'

data = pd.read_csv(filename).fillna(0)
data.set_index('Unnamed: 0', inplace=True)
data = data.transpose()
y = data[classname]
X = data.drop([classname], axis=1)

selector = VarianceThreshold()
selector.fit_transform(X)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit_transform(X)
#pt = PowerTransformer()
#pt.fit_transform(X)

ind = k1_best(X, y)
X = X.iloc[:, ind]
model = MMCCModel()
run_select_and_test(model.select_features, X, y)







