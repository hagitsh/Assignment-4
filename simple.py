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
filename = '../bioconductor/breastCancerVDX.csv'
classname = 'oestrogenreceptorsClass'
filename = '../bioconductor/COPDSexualDimorphism.data.csv'
classname = 'DiagClass'
filename = '../bioconductor/curatedOvarianData.csv'
classname = 'GradeClass'
filename = '../bioconductor/leukemiasEset.csv'
classname = 'LeukemiaTypeClass'


data = pd.read_csv(filename).fillna(0)
data.set_index('Unnamed: 0', inplace=True)
data = data.transpose()
y = data[classname]
X = data.drop([classname], axis=1)
original_n_features = X.shape[1]

selector = VarianceThreshold()
selector.fit_transform(X)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit_transform(X)
#pt = PowerTransformer()
#pt.fit_transform(X)

ind = k1_best(X, y)
X = X.iloc[:, ind]
model = MMCCModel()
with open('results.csv', 'w') as results_file:
    run_select_and_test(reliefF, X, y, results_file, filename.split('/')[-1], original_n_features, 'reliefF')
    run_select_and_test(rfe, X, y, results_file, filename.split('/')[-1], original_n_features, 'rfe')
    run_select_and_test(Fdr_f_classif, X, y, results_file, filename.split('/')[-1], original_n_features, 'Fdr_f_classif')
    run_select_and_test(mrmr, X, y, results_file, filename.split('/')[-1], original_n_features, 'mrmr')
    run_select_and_test(model.select_features, X, y, results_file, filename.split('/')[-1], original_n_features, 'MMCC')








