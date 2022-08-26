from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, average_precision_score
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

MODEL_LIST = {"SVM": LinearSVC, "k - nearest": NearestCentroid, "RandomForest": RandomForestClassifier,
              "LogisticsRegression": LogisticRegression, "NB": GaussianNB}
FEATURES_NUMBERS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

def test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc = 0
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test.values, y_prob[:, 1])
    except:
        pass
    return {
        "acc": accuracy_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "auc": auc,
        "auc_pr": average_precision_score(y_test, y_pred)
    }


def run_select_and_test(selection_method, X, y):
    counter = 0
    results = {}
    featues = selection_method(X, y)
    Xo = X.copy()
    for n_features in FEATURES_NUMBERS:
        results[n_features] = {}
        X = Xo.iloc[:, featues[0:n_features]].copy()
        for model_name in MODEL_LIST:
            model = MODEL_LIST[model_name]
            results[n_features][model_name] = []
            if len(y) < 50:
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
    for n_features in FEATURES_NUMBERS:
        print(f"number of features {n_features}")
        for model_name in MODEL_LIST:
            print(model_name)
            results_sum = {
                "acc": 0.,
                "mcc": 0.,
                "auc": 0.,
                "auc_pr": 0.
            }
            for r in results[n_features][model_name]:
                for p in r:
                    results_sum[p] += r[p] / len(results[n_features][model_name])
            print(results_sum)
