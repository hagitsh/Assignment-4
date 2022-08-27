from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

MODEL_LIST = {"SVM": LinearSVC}
FEATURES_NUMBERS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

def test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc = 0
    auc_pr = 0
    try:
        auc_pr = average_precision_score(y_test, y_pred)
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test.values, y_prob[:, 1])
    except:
        pass
    return {
        "acc": "{:.2f}".format(accuracy_score(y_test, y_pred)),
        "mcc": "{:.2f}".format(matthews_corrcoef(y_test, y_pred)),
        "auc": "{:.2f}".format(auc),
        "auc_pr": "{:.2f}".format(auc_pr)
    }

def run_select_and_test(selection_method, X, y, results_file, dataset_name, original_n_features, selection_methomd_name):
    counter = 0
    results = {}
    featues = selection_method(X, y)
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
                for i in range(len(y)):
                    counter += 1
                    if counter % 100 == 0:
                        print(counter)
                    for j in range(i + 1, len(y)):
                        X_test = X.iloc[[i, j], :]
                        y_test = y.iloc[[i, j]]
                        train = list(set(range(len(y))) - {i, j})
                        X_train = X.iloc[train, :]
                        y_train = y.iloc[train]
                        results[n_features][model_name].append(test_model(model(), X_train, y_train, X_test, y_test))
            elif len(y) < 100:
                cv_method = 'leave one out'
                for i in range(len(y)):
                    counter += 1
                    if counter % 100 == 0:
                        print(counter)
                    X_test = X.iloc[[i], :]
                    y_test = y.iloc[[i]]
                    train = list(set(range(len(y))) - {i})
                    X_train = X.iloc[train, :]
                    y_train = y.iloc[train]
                    results[n_features][model_name].append(test_model(model(), X_train, y_train, X_test, y_test))
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
                    results[n_features][model_name].append(test_model(model(), X_train, y_train, X_test, y_test))

    for n_features in FEATURES_NUMBERS:
        print(n_features)
        for model_name in MODEL_LIST:
            print(model_name)
            fold = 0
            for f in results[n_features][model_name]:
                fold += 1
                for p in f:
                    results_file.write(f"{dataset_name},{len(y)},{original_n_features},{selection_methomd_name},{model_name},{n_features},{cv_method},{fold},{p},{f[p]},{feature_names[n_features][model_name]}\n")
