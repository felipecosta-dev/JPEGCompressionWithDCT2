import numpy as np
import matplotlib.pyplot as plt
import time
from manualdct2 import ManualDCT2
from scipydct2 import SciPyDCT2

def generate_matrix(N):
    return np.random.rand(N, N)

def measure_time(dct_function, matrix):
    start_time = time.time()
    dct_function(matrix)
    end_time = time.time()
    return end_time - start_time

def print_matrix(matrix, name):
    print(f"{name}:")
    for row in matrix:
        print(" ".join(f"{val:.2e}" for val in row))
    print()

def main():
    sizes = [8, 16, 32, 64, 128]
    manual_times = []
    scipy_times = []
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

    for size in sizes:
        matrix = generate_matrix(size)
        manual_dct = ManualDCT2(block_size=size)
        scipy_dct = SciPyDCT2()

        # Measure manual DCT2 time
        manual_time = measure_time(manual_dct.dct2, matrix)
        manual_times.append(manual_time)

        # Measure SciPy DCT2 time
        scipy_time = measure_time(scipy_dct.dct2, matrix)
        scipy_times.append(scipy_time)

        print(f"Size: {size}x{size}, Manual DCT Time: {manual_time:.6f} s, SciPy DCT Time: {scipy_time:.6f} s")

    # Plot results
    plt.figure()
    plt.plot(sizes, manual_times, label='Manual DCT', marker='o')
    plt.plot(sizes, scipy_times, label='SciPy DCT', marker='o')
    plt.yscale('log')
    plt.xlabel('Matrix Size (NxN)')
    plt.ylabel('Time (s)')
    plt.title('DCT Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
