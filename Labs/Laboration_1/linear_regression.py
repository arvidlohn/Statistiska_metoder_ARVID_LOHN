import numpy as np
import scipy.stats as stats


class Linear_Regression():
    
    def __init__(self, x, y):
        self.d = None
        self.n = None
        self.beta = None
        self.x = x
        self.y = y

        #Felhantering
        if self.x.ndim != 2:
            raise ValueError(
            f"x must be 2 dimensional. Current sharp is {self.x.shape}")
    

    def least_squared_approximation(self):
        b = np.linalg.pinv(self.x.T @ self.x)
        self.beta = b @ self.x.T @ self.y
        return self
    
    def dd (self):

        self.d = self.x.shape[1]
        return self
    
    def nn (self):
        self.n = self.x.shape[0]
        return self
    
    def calculate_var(self):
        var = self.calculate_SSE() / (self.n - self.d - 1)
        return var
    
    def calculate_SSE(self):
        y_hat = self.x @ self.beta
        rest = self.y - y_hat
        sse = (rest**2).sum()
        return sse
    
    def standard_deviation(self):
        return np.sqrt(self.calculate_var())
    
    def calculate_RMSE(self):
        rmse = np.sqrt(self.calculate_SSE() / self.n)
        return rmse