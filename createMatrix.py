import numpy as np

class Matrix:
    def __init__(self):
        self.m, self.n = 10,10
        self.base_P = None


    def generateMatrix(self):
        np.random.seed(44)
        self.base_P = np.random.randint(-1, 2, size=(self.m, self.n))  # integer values in [-1, 1]
        return self.base_P

    def returnSize(self):
        return (self.m, self.n)
