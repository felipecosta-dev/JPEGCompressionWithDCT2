import math
import numpy as np

class ManualDCT2:
    def __init__(self, N):
        self.N = N
        self.cos_table = self.generate_cos_table(N)

    def generate_cos_table(self, N):
        return [
            [math.cos(math.pi * k * (2 * i + 1) / (2 * N)) for k in range(N)] for i in range(N)
        ]
    
    def get_scaling_factor(self, k, n):
        if k == 0:
            scaling = math.sqrt(1 / (4 * n))
        else:
            scaling = math.sqrt(1 / (2 * n))
    
        return scaling

    def dct1d(self, a):
        n = len(a)
        output = np.zeros(n)

        for k in range(n):
            for i in range(n):
                output[k] += a[i] * self.cos_table[i][k]
            output[k] *= 2 * self.get_scaling_factor(k, n)

        return output

    def dct2(self, a):
        M, N = a.shape
        output = np.empty([M, N])

        # Apply DCT1D to each row
        for i in range(M):
            output[i, :] = self.dct1d(a[i, :])

        # Apply DCT1D to each column of the intermediate result
        for j in range(N):
            output[:, j] = self.dct1d(output[:, j])

        return output
