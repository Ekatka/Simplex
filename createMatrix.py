import numpy as np

#TODO: testy treninku vuci velikosti matic

class Matrix:
    def __init__(self):
        self.m, self.n = 5,5
        self.base_P = None
        self.epsilon = 0.1
        np.random.seed(46)

    def resize(self, new_m, new_n):
        self.m = new_m
        self.n = new_n


    def generateMatrix(self):
        self.base_P = np.random.randint(-1, 2, size=(self.m, self.n))  # integer values in [-1, 1]
        return self.base_P

    def returnSize(self):
        return (self.m, self.n)

    def returnEpsilon(self):
        return self.epsilon
