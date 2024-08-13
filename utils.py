import numpy as np

def apply_frequency_cutoff(dct_block, d):
    F = dct_block.shape[0]
    for k in range(F):
        for l in range(F):
            if k + l >= d:
                dct_block[k, l] = 0
    return dct_block
