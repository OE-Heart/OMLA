import numpy as np

class Perceptron:
    def __init__(self, X, y):
        self.w = np.ones(len(X[0]), dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        self.X_train = X
        self.y_train = y

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return 1 if y >= 0 else -1

    # SGD
    def fit(self):
        while True:
            has_wrong = False
            for i in range(len(self.X_train)):
                X = self.X_train[i]
                y = self.y_train[i]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w += self.l_rate * np.dot(y, X)
                    self.b += self.l_rate * y
                    has_wrong = True
            if not has_wrong:
                break
        return 'SGD done!'