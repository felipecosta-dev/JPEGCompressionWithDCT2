import numpy as np
import matplotlib.pyplot as plt
import time
from manualdct2 import ManualDCT2
from scipydct2 import SciPyDCT2

def print_matrix(matrix, name):
    print(f"{name}:")
    for row in matrix:
        print(" ".join(f"{val:.2e}" for val in row))
    print()

def main():
    test_block = np.array([
        [231, 32, 233, 161, 24, 71, 140, 245],
        [247, 40, 248, 245, 124, 204, 36, 107],
        [234, 202, 245, 167, 9, 217, 239, 173],
        [193, 190, 100, 167, 43, 180, 8, 70],
        [11, 24, 210, 177, 81, 243, 8, 112],
        [97, 195, 203, 47, 125, 114, 165, 181],
        [193, 70, 174, 167, 41, 30, 127, 245],
        [87, 149, 57, 192, 65, 129, 178, 228]
    ])

    # Manual DCT2
    manual_dct = ManualDCT2(N=8)
    manual_dct_result = manual_dct.dct2(test_block)

    # SciPy DCT2
    scipy_dct = SciPyDCT2()
    scipy_dct_result = scipy_dct.dct2(test_block)

    # Printing Results
    print_matrix(test_block, "Original Block")
    print_matrix(manual_dct_result, "Manual DCT2 Result")
    print_matrix(scipy_dct_result, "SciPy DCT2 Result")

    # Test the first row with 1D DCT
    first_row = test_block[0, :]
    manual_dct_1d_result = manual_dct.dct1d(first_row)
    scipy_dct_1d_result = scipy_dct.dct2(first_row)

    print("First Row Transformation:")
    print("Original:", first_row)
    print("Manual DCT1D:", manual_dct_1d_result)
    print("SciPy DCT1D:", scipy_dct_1d_result)

if __name__ == "__main__":
    main()
