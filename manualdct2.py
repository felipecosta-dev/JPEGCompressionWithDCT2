import math
import numpy as np

class ManualDCT2:
    def __init__(self, N):
        """Initialize the DCT calculator for an NxN block."""
        self.N = N
        self.cos_table = self.generate_cos_table(N)
        self.range_list = self.generate_range_list(N)
        self.root2_inv = 1 / math.sqrt(2)

    def generate_cos_table(self, N):
        """Generate a cosine table for size N."""
        return [
            [math.cos((2 * i + 1) * j * math.pi / (2 * N)) for j in range(N)] for i in range(N)
        ]

    def generate_range_list(self, N):
        """Generate a range list for size N."""
        return [(i, j) for i in range(N) for j in range(N)]

    def compute_dct(self, a, u, v):
        """Compute the DCT coefficient for position (u, v) in an N x N block."""
        r = 0
        for i, j in self.range_list:
            r += a[i][j] * self.cos_table[i][u] * self.cos_table[j][v]
        
        if u == 0: r *= self.root2_inv
        if v == 0: r *= self.root2_inv
        r *= 0.25 if self.N == 8 else (2 / self.N)
        
        return r

    def dct2(self, a):
        """Compute the 2D DCT for an NxN matrix."""
        dct_block = np.zeros((self.N, self.N))

        for u in range(self.N):
            for v in range(self.N):
                dct_block[u, v] = self.compute_dct(a, u, v)
        
        return dct_block
    
    def dct1d(self, vector):
        N = len(vector)
        result = np.zeros(N)

        for k in range(N):
            alpha_k = self.root2_inv if k == 0 else 1
            result[k] = alpha_k * sum(vector[i] * math.cos((2 * i + 1) * k * math.pi / (2 * N)) for i in range(N))
        
        return result
