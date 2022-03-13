import numpy as np
import math

class NaiveBayes:
    def __init__(self) -> None:
        self.model = None

    def mean(self, X):
        return float(sum(X)) / len(X)

    def var(self, X):
        mean = self.mean(X)
        return float(sum([pow(x - mean, 2) for x in X])) / len(X)

    def stdev(self, X):
        var = self.var(X)
        return math.sqrt(var)

    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def summarize(self, train_data):
        return [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]

    def fit(self, X_train, y_train):
        labels = list(set(y_train))
        data = {label: [] for label in labels}
        for X, label in zip(X_train, y_train):
            data[label].append(X)
        self.model = {
            label: self.summarize(value) for label, value in data.items()
        }
        return 'NaiveBayes()'

    def calc(self, X):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(X[i], mean, stdev)
        return probabilities

    def predict(self, X):
        label = sorted(self.calc(X).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right_cnt = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_cnt += 1
        return float(right_cnt) / len(X_test)