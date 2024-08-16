import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from manualdct2 import ManualDCT2
from scipydct2 import SciPyDCT2

pd.options.display.float_format = '{:.10f}'.format
fig_save_dir = r'G:\My Drive\Facultad\UniMiB\2023-2024\Metodi del calcolo scientifico\Progetto1\Report\images'

def generate_matrix(N):
    return np.random.rand(N, N)

def measure_time(dct_function, matrix):
    start_time = time.perf_counter()
    dct_function(matrix)
    end_time = time.perf_counter()
    return end_time - start_time

def print_matrix(matrix, name):
    print(f"{name}:")
    for row in matrix:
        print(" ".join(f"{val:.2e}" for val in row))
    print()

def main():
    sizes = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64]
    results = []

    for size in sizes:
        matrix = generate_matrix(size)
        manual_dct = ManualDCT2(N=size)
        scipy_dct = SciPyDCT2()

        manual_time = measure_time(manual_dct.dct2, matrix)
        scipy_time = measure_time(scipy_dct.dct2, matrix)

        n_squared = size ** 2
        n_cubed = size ** 3
        n_squared_log_n = size ** 2 * np.log(size)

        results.append({
            "N": size,
            "Manual Time (s)": manual_time,
            "SciPy Time (s)": scipy_time,
            "N^2": n_squared,
            "N^3": n_cubed,
            "N^2 * log(N)": n_squared_log_n
        })

    df = pd.DataFrame(results)

    # Save the table to a CSV file
    df.to_csv("dct_performance_results.csv", index=False)
    print("\nTable of Results:")
    print(df)

    # Plot results
    plt.figure()
    plt.plot(sizes, df["Manual Time (s)"], label='Manual DCT', marker='o')
    plt.plot(sizes, df["SciPy Time (s)"], label='SciPy DCT', marker='o')
    plt.yscale('log')
    plt.xlabel('Matrix Size (NxN)')
    plt.ylabel('Time (s)')
    plt.title('DCT Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(fig_save_dir), f'part1.png'))

if __name__ == "__main__":
    main()
