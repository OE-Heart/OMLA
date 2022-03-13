from cProfile import label
from itertools import count
from turtle import right
import numpy as np
from collections import Counter

class kNN:
    def __init__(self, X_train, y_train, k=3, p=2) -> None:
        self.k = k  # 临近点个数
        self.p = p  # 距离度量
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        knn_list = []
        for i in range(self.k):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.k, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        knn = [k[-1] for k in knn_list]
        counter = Counter(knn)
        label = sorted(counter.items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right_cnt = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_cnt += 1
        return right_cnt / len(X_test)