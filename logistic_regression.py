import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, tolerance = 1e-6, learning_rate = 0.01, n_steps = 1000):
        self.tolerance = tolerance
        self.n_steps = n_steps
        self.learning_rate = learning_rate

        self.w = None
        self.b = None

        self.loss = []

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def bce_loss(y_true, p_hat):
        return -np.mean(y_true * np.log(p_hat) + (1- y_true) *  np.log(1-p_hat))

    
    def fit(self,X,y):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        n_samples,n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0
        prev_loss = float('inf')

        for i in tqdm(range(self.n_steps)):

            p_hat = self.sigmoid(X @ self.w + self.b)

            error = p_hat - y

            loss = self.bce_loss(y, p_hat)

            self.loss.append(loss)

            abs_loss_diff = abs(prev_loss - loss)

            if abs_loss_diff < self.tolerance:
                print(f" Converged at iteration {i}, Final Loss: {loss:.6f}")
                break

            prev_loss = loss

            dw = (1/n_samples) * X.T @ error
            db = (1/n_samples) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            return self

        def predict_proba(self,X):
            if isinstance(X,pd.DataFrame):
                X = X.to_numpy()

            return self.sigmoid(X @ self.w + self.b)

        def predict(self, X):
            p_hat = self.predict_proba(X)

            return (p_hat > 0.5).astype(int)

        def score(self, X, y):
            if isinstance(y, pd.Series):
                y = y.to_numpy()
            y_pred = self.predict(X)

            return np.mean(y_pred == y)