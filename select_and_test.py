import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import time

MODEL_LIST = {"SVM": LinearSVC, "k - nearest": KNeighborsClassifier, "RandomForest": RandomForestClassifier,
               "LogisticsRegression": LogisticRegression, "NB": GaussianNB}
FEATURES_NUMBERS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]


def test_model(model, X_train, y_train, X_test, y_test, y_testc, durations):
    start = time.time()
    try:
        clf = CalibratedClassifierCV(model)
        clf.fit(X_train, y_train)
        model = clf
    except:
        model.fit(X_train, y_train)
    durations['train_duration'] = "{:.3f}".format((time.time() - start))
    start = time.time()
    y_pred = model.predict(X_test)
    durations['test_duration'] = "{:.3f}".format((time.time() - start))
    y_prob = model.predict_proba(X_test)
    classes_in_test = len(np.unique(y_test))
    if classes_in_test > 1:
        num_classes = len(model.classes_)
        if len(model.classes_) > 2:
            auc_pr_i = []
            auc_i = []
            for c in range(num_classes):
                auc_pr_i.append(average_precision_score(y_testc[:, c], y_prob[:, c]))
                if len(np.unique(y_testc[:, c])) > 1 and classes_in_test < num_classes:
                    auc_i.append(roc_auc_score(y_testc[:, c], y_prob[:, c]))
            auc_pr = np.mean(auc_pr_i)
        else:
            auc_pr = average_precision_score(y_testc[:,1], y_prob[:,0])
        if classes_in_test > 2:
            if classes_in_test == num_classes:
                auc = roc_auc_score(y_test.values, y_prob, multi_class='ovr')
            else:
                auc = np.mean(auc_i)
        else:
            auc = roc_auc_score(y_test.values, y_prob[:, 0])
        auc = "{:.2f}".format(auc)
        return {
            "acc": "{:.2f}".format(accuracy_score(y_test, y_pred)),
            "mcc": "{:.2f}".format(matthews_corrcoef(y_test, y_pred)),
            "auc": auc,
            "auc_pr": "{:.2f}".format(auc_pr)
        }
    else:
        return {
            "acc": "{:.2f}".format(accuracy_score(y_test, y_pred)),
            "mcc": "{:.2f}".format(matthews_corrcoef(y_test, y_pred)),
            "y_test": y_test.values[0],
            "y_testc": y_testc[0],
            "y_pred": y_prob[0],
            "classes": model.classes_
        }




def run_select_and_test(selection_method, X, y, results_file, dataset_name, original_n_features,
                        selection_methomd_name, time_results_file):
    yc = to_categorical(y)
    loo = LeaveOneOut()
    lpo = LeavePOut(2)
    counter = 0
    results = {}
    start = time.time()
    featues = selection_method(X, y)
    duration = "{:.3f}".format((time.time() - start))
    time_results_file.write(f"{dataset_name},{selection_methomd_name},selection runtime, {duration}\n")
    feature_names = {}
    Xo = X.copy()
    for n_features in FEATURES_NUMBERS:
        results[n_features] = {}
        feature_names[n_features] = {}
        X = Xo.iloc[:, featues[0:n_features]].copy()
        for model_name in MODEL_LIST:
            model = MODEL_LIST[model_name]
            results[n_features][model_name] = []
            feature_names[n_features][model_name] = list(X.columns)
            if len(y) < 50:
                cv_method = 'leave pair out'
                for train_index, test_index in lpo.split(X):
                    counter += 1
                    if counter % 100 == 0:
                        print(counter)
                    X_train = X.iloc[train_index, :]
                    y_train = y.iloc[train_index]
                    X_test = X.iloc[test_index, :]
                    y_test = y.iloc[test_index]
                    durations = {}
                    y_testc = yc[test_index]
                    results[n_features][model_name].append(
                        test_model(model(), X_train, y_train, X_test, y_test, y_testc, durations))
            elif len(y) < 100:
                cv_method = 'leave one out'
                for train_index, test_index in loo.split(X):
                    counter += 1
                    if counter % 100 == 0:
                        print(counter)
                    X_train = X.iloc[train_index, :]
                    y_train = y.iloc[train_index]
                    X_test = X.iloc[test_index, :]
                    y_test = y.iloc[test_index]
                    durations = {}
                    y_testc = yc[test_index]
                    results[n_features][model_name].append(
                        test_model(model(), X_train, y_train, X_test, y_test, y_testc, durations))
            else:
                n_splits = 10 if len(y) < 1000 else 5
                cv_method = f"{n_splits} folds"
                kf = KFold(n_splits=n_splits, shuffle=True)
                for train_index, test_index in kf.split(X):
                    counter += 1
                    if counter % 100 == 0:
                        print(counter)
                    X_test = X.iloc[test_index, :]
                    y_test = y.iloc[test_index]
                    X_train = X.iloc[train_index, :]
                    y_train = y.iloc[train_index]
                    durations = {}
                    y_testc = yc[test_index]
                    results[n_features][model_name].append(
                        test_model(model(), X_train, y_train, X_test, y_test, y_testc, durations))
            time_results_file.write(
                f"{dataset_name},model {model_name} with {n_features} features, train duration, {durations['train_duration']}\n")
            time_results_file.write(
                f"{dataset_name},model {model_name} with {n_features} features, test duration, {durations['test_duration']}\n")
    for n_features in FEATURES_NUMBERS:
        print(n_features)
        for model_name in MODEL_LIST:
            print(model_name)
            fold = 0
            y_pred = []
            y_test = []
            classes = []
            y_testc = []
            for f in results[n_features][model_name]:
                fold += 1
                if "auc" in f:
                    for p in f:
                        pass
                        results_file.write(f"{dataset_name},{len(y)},{original_n_features},{selection_methomd_name},{model_name},{n_features},{cv_method},{fold},{p},{f[p]},{str(feature_names[n_features][model_name]).replace(',', ';')}\n")
                elif 'y_pred' in f:
                    results_file.write(
                        f"{dataset_name},{len(y)},{original_n_features},{selection_methomd_name},{model_name},{n_features},{cv_method},{fold},'acc',{f['acc']},{str(feature_names[n_features][model_name]).replace(',', ';')}\n")
                    results_file.write(
                        f"{dataset_name},{len(y)},{original_n_features},{selection_methomd_name},{model_name},{n_features},{cv_method},{fold},'mcc',{f['mcc']},{str(feature_names[n_features][model_name]).replace(',', ';')}\n")
                    y_pred.append(f['y_pred'])
                    y_testc.append(f['y_testc'])
                    y_test.append(f['y_test'])
                    classes = f['classes']
            if len(y_pred) > 0:
                y_pred = np.array(y_pred)
                y_testc = np.array(y_testc)
                num_classes = len(classes)
                if num_classes > 2:
                    classes_in_test = len(np.unique(y_test))
                    if classes_in_test == num_classes:
                        auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
                    auc_i = []
                    auc_pr_i = []
                    for c in range(num_classes):
                        auc_pr_i.append(average_precision_score(y_testc[:, c], y_pred[:, c]))
                        if len(np.unique(y_testc[:, c])) > 1 and classes_in_test < num_classes:
                            auc_i.append(roc_auc_score(y_testc[:, c], y_pred[:, c]))
                    if classes_in_test < num_classes:
                        auc = np.mean(auc_i)
                    auc_pr = np.mean(auc_pr_i)
                else:
                    try:
                        auc = roc_auc_score(y_test, y_pred[:, 1])
                    except:
                        auc = 0
                    y_test = to_categorical(y_test)
                    try:
                        auc_pr = average_precision_score(y_test[:, 1], y_pred[:, 0])
                    except:
                        auc_pr = 0

                results_file.write(
                    f"{dataset_name},{len(y)},{original_n_features},{selection_methomd_name},{model_name},{n_features},{cv_method},{fold},'auc',{auc},{str(feature_names[n_features][model_name]).replace(',', ';')}\n")
                results_file.write(
                    f"{dataset_name},{len(y)},{original_n_features},{selection_methomd_name},{model_name},{n_features},{cv_method},{fold},'auc_pr',{auc_pr},{str(feature_names[n_features][model_name]).replace(',', ';')}\n")
