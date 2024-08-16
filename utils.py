import numpy as np

def apply_frequency_cutoff(dct_block, d):
    F = dct_block.shape[0]

    if d == (2 * F - 2):
        dct_block[F - 1, F - 1] = 0
        return dct_block

    if d == 0:
        return np.zeros_like(dct_block)
    
    for k in range(F):
        for l in range(F):
            if k + l >= d:
                dct_block[k, l] = 0

    return dct_block
