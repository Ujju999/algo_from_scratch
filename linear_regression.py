import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self,learning_rate = 0.01,tolerance = 1e-6, n_steps = 1000):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.n_steps = n_steps

        self.w = None
        self.b = None

        self.loss = []
    
    def fit(self,X,y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y,pd.Series):
            y = y.to_numpy()

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0
        prev_loss = float('inf')

        for i in range(self.n_steps):
            y_hat = X @ self.w + self.b
            error = y_hat - y

            loss = np.mean(np.square(error))

            abs_loss_diff = abs(loss - prev_loss)

            self.loss.append(loss)

            if abs_loss_diff < self.tolerance:
                print(f"Converged at iteration {i}, Final Loss : {loss:.6f}")
                break

            dw, db = (2/n_samples) * X.T @ error, (2/n_samples) * np.sum(error)

            self.w = self.w  - self.learning_rate * dw
            self.b = self.b  - self.learning_rate * db

            if (i + 1) % (self.n_steps // 10) == 0:
                print(f"Iteration {i+1}: Loss = {loss:.6f}")

            prev_loss = loss

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        y_hat = X @ self.w + self.b
        return y_hat

    def score(self,X,y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y,pd.Series):
            y = y.to_numpy()
        
        y_hat = self.predict(X)
        ss_residual = np.sum(np.square(y_hat - y))
        ss_total = np.sum(np.square(y - np.mean(y)))

        r2= 1 - np.divide(ss_residual,ss_total)
        return r2

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.random(1000,2) * 10
    y = 3 * X.squeeze() + 7 + np.random.randn(100) * 0.5

    model = LinearRegression(learning_rate=0.001,tolerance=1e-6,n_steps=10000)
    model.fit(X,y)
    print("Complete")
