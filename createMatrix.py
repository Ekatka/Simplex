import numpy as np

#TODO: testy treninku vuci velikosti matic

class Matrix:
    def __init__(self, m=5, n=5, min=-1, max=1):
        self.m, self.n = m,n
        self.min, self.max = min, max
        self.base_P = None
        self.epsilon = 0.1
        np.random.seed(46)

    def resize(self, new_m, new_n):
        self.m = new_m
        self.n = new_n

    def generateMatrix(self):
        self.base_P = np.random.randint(self.min, self.max+1, size=(self.m, self.n))  # integer values in [-1, 1]
        return self.base_P

    def returnSize(self):
        return (self.m, self.n)

    def returnEpsilon(self):
        return self.epsilon
