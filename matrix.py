import numpy as np

class Matrix:
    # matrix initialization
    def __init__(self, m=3, n=3, min=-1, max=1, epsilon=0.1, base_P=None):
        self.m, self.n = m,n
        self.min, self.max = min, max
        self.base_P = base_P
        self.epsilon = epsilon

    def resize(self, new_m, new_n):
        self.m = new_m
        self.n = new_n

    # generating matrix according to params
    def generateMatrix(self):
        self.base_P = np.random.randint(self.min, self.max + 1, size=(self.m, self.n))
        self.base_P = self.base_P*10
        return self.base_P

    def returnSize(self):
        return (self.m, self.n)

    def returnEpsilon(self):
        return self.epsilon

    # add epsilon noise to matrix
    def generate_perturbed_matrix(self):
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=(self.m, self.n))
        noise = np.round(noise, decimals=4)
        perturbed_P = self.base_P + noise
        return Matrix(self.m, self.n, self.min, self.max, epsilon=0, base_P=perturbed_P)

    def copy(self):
        return Matrix(self.m, self.n, self.min, self.max, self.epsilon, np.copy(self.base_P))

