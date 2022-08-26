import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.optimize import differential_evolution


class MMCC:
    def __init__(self, s, b, X, y, count):
        self.y = y
        self.X = X
        self.s = s
        self.b = b
        self.count = count

    def correntropy2(self, x2):
        return lambda x1: np.average(np.exp(-0.5 * (x1 - x2) ** 2 / self.s))

    def correntropy(self, x):
        self.count = self.count + 1
        if self.count % 100 == 0:
            print(self.count)
        V = np.average(np.exp(-0.5 * (x - self.y) ** 2 / self.s))
        if np.isnan(V):
            V = 0
        bV = np.apply_along_axis(self.correntropy2(x), 0, self.X.values)
        bV = bV[~np.isnan(bV)].sum()
        return V - self.b * bV

    def run(self, X):
        return np.apply_along_axis(self.correntropy, 0, X.values)


class MMCCModel:
    def __init__(self):
        self.count = 0

    def run_mmcc(self, params):
        mmcc = MMCC(params[0], params[1], self.X, self.y, self.count)
        self.count += self.X.shape[1]
        return np.argpartition(mmcc.run(self.X), -self.num_features)[-self.num_features:]

    def eval(self, params):
        ind = self.run_mmcc(params)
        model = LinearSVC()
        X_train, X_test, y_train, y_test = train_test_split(self.X.iloc[:, ind].values, self.y.values, train_size=0.8,
                                                            shuffle=True)
        model.fit(X_train, y_train)
        return -model.score(X_test, y_test)

    def select_features(self, X, y, num_features=100):
        self.X = X
        self.y = y
        self.num_features = num_features
        results = differential_evolution(self.eval, [(1, 3), (1, 3)], maxiter=2)
        print(results)
        return self.run_mmcc(results.x)

