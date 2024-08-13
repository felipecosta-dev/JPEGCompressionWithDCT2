import numpy as np

class ManualDCT2:
    def __init__(self, block_size):
        self.block_size = block_size
        self.dct_matrix = self.create_dct_matrix(block_size)
        self.idct_matrix = self.dct_matrix.T

    def create_dct_matrix(self, N):
        dct_matrix = np.zeros((N, N))
        for k in range(N):
            for l in range(N):
                alpha_k = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
                alpha_l = np.sqrt(1 / N) if l == 0 else np.sqrt(2 / N)
                for i in range(N):
                    for j in range(N):
                        dct_matrix[k, l] += (np.cos((2 * i + 1) * k * np.pi / (2 * N)) * 
                                             np.cos((2 * j + 1) * l * np.pi / (2 * N)))
                dct_matrix[k, l] *= alpha_k * alpha_l
        return dct_matrix

    def dct2(self, block):
        return np.dot(self.dct_matrix, np.dot(block, self.dct_matrix.T))

    def idct2(self, block):
        return np.dot(self.idct_matrix, np.dot(block, self.idct_matrix.T))

# Example usage:
block = np.array([
    [231, 32, 233, 161, 24, 71, 140, 245],
    [247, 40, 248, 245, 124, 204, 36, 107],
    [234, 202, 245, 167, 9, 217, 239, 173],
    [193, 190, 100, 167, 43, 180, 8, 70],
    [11, 24, 210, 177, 81, 243, 8, 112],
    [97, 195, 203, 47, 125, 114, 165, 181],
    [193, 70, 174, 167, 41, 30, 127, 245],
    [87, 149, 57, 192, 65, 129, 178, 228]
])

manual_dct = ManualDCT2(block_size=8)
dct_block = manual_dct.dct2(block)
idct_block = manual_dct.idct2(dct_block)
print("DCT Block:\n", dct_block)
print("IDCT Block:\n", idct_block)
