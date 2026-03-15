import numpy as np
import time
import matplotlib.pyplot as plt
import rand_vs_det_new as rvd  

# --- Configuration & Constants ---
MATRIX_ROWS = 1000
MATRIX_COLS = 1000
SEED = 42

def generate_low_rank_matrix(m, n, rank=200):
    """
    Generates a large matrix with rapidly decaying singular values 
    to simulate real-world low-rank data.
    """
    np.random.seed(SEED)
    # Generate orthogonal matrices
    U = np.linalg.qr(np.random.randn(m, rank))[0]
    V = np.linalg.qr(np.random.randn(n, rank))[0]
    
    # Rapidly decaying singular values from 1e4 down to 1e-3
    sing_vals = np.geomspace(1e4, 1e-3, num=rank) 
    
    # Construct A = U * Sigma * V^T
    Sigma = np.diag(sing_vals)
    A = U @ Sigma @ V.T
    return A

def generate_gaussian_matrix(m, n):
    """Returns a Gaussian random matrix with fixed seed for reproducibility."""
    np.random.seed(SEED)
    return np.random.randn(m, n)

def calculate_reconstruction_error(A, U, S, Vt):
    """
    Computes the relative Frobenius norm error between the original matrix A 
    and its low-rank approximation.
    """
    norm_A = np.linalg.norm(A, 'fro')
    # Reconstruct approximation
    A_approx = U @ np.diag(S) @ Vt
    
    norm_error = np.linalg.norm(A - A_approx, 'fro')
    relative_error_percent = (norm_error / norm_A) * 100
    return relative_error_percent

def run_svd_experiment(algorithm_func, A, n_components, **kwargs):
    """
    Helper function to run a single SVD execution and measure time/error.
    """
    start_time = time.time()
    U, S, Vt = algorithm_func(A, n_components=n_components, **kwargs)
    duration = time.time() - start_time
    
    error = calculate_reconstruction_error(A, U, S, Vt)
    return duration, error

def run_complexity_analysis(A, algorithm_func, param_list, x_metric='rank', **kwargs):
    """
    Runs a series of tests iterating over a parameter (e.g., rank or iterations).
    
    :param x_metric: 'rank' (x-axis = n_components) or 'time' (x-axis = duration)
    """
    x_values = []
    y_errors = []
    
    print(f"Running analysis for {algorithm_func.__name__}...")
    
    for param in param_list:
        # Determine if we are iterating over components or iterations
        if 'max_iter' in kwargs and isinstance(kwargs['max_iter'], list):
             pass
        
        # Execute SVD
        duration, error = run_svd_experiment(algorithm_func, A, n_components=param, **kwargs)
        
        if x_metric == 'time':
            current_time = duration
            if x_values and current_time <= x_values[-1]:
                current_time = x_values[-1] + 0.001
            x_values.append(current_time)
        else:
            x_values.append(param)
            
        y_errors.append(error)
        
        if error < 1e-2: 
            break
            
    return x_values, y_errors

def run_time_comparison_by_rank(A, ranks, det_kwargs=None, rand_kwargs=None):
    """
    Measures wall-clock time for deterministic vs randomized SVD over ranks.
    Returns two lists aligned with the provided ranks.
    """
    det_kwargs = det_kwargs or {}
    rand_kwargs = rand_kwargs or {}
    det_times = []
    rand_times = []

    for k in ranks:
        det_duration, _ = run_svd_experiment(rvd.deterministic_svd, A, n_components=k, **det_kwargs)
        rand_duration, _ = run_svd_experiment(rvd.rand_svd, A, n_components=k, **rand_kwargs)
        det_times.append(det_duration)
        rand_times.append(rand_duration)

    return det_times, rand_times


def plot_results(data_series, title, xlabel, ylabel="Reconstruction Error (%)"):
    """
    Generic plotting function.
    data_series: List of tuples (x_values, y_values, label_string)
    """
    plt.figure(figsize=(10, 6))
    for x, y, label in data_series:
        plt.plot(x, y, label=label, marker='o', markersize=3)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

# --- Main Benchmark Routine ---
def run_full_benchmark():
    print("--- Initializing Benchmark Suite ---")
    
    # 1. Data Generation
    print("Generating Synthetic Matrix...")
    A = generate_low_rank_matrix(MATRIX_ROWS, MATRIX_COLS)
    # Optional: Gaussian Matrix
    # A_gauss = np.random.randn(MATRIX_ROWS, MATRIX_COLS)
    
    print(f"Matrix Shape: {A.shape}")

    # 2. Single Point Comparisons (Sanity Check)
    check_ranks = [300, 15, 1]
    print("\n--- Single Point Checks (Randomized SVD) ---")
    for k in check_ranks:
        dur, err = run_svd_experiment(rvd.rand_svd, A, n_components=k)
        print(f"k={k}: {dur:.4f}s | Error: {err:.4f}%")
    """
    # 3. Experiment: Deterministic SVD (Error vs Time)
    # Comparing different max_iter settings for the deterministic alg (if applicable)
    # Assuming deterministic_svd takes max_iter, otherwise remove kwargs
    print("\n--- Experiment: Deterministic SVD Performance ---")
    ranks_to_test = range(1, 1100, 1)
    
    # Run 1: 1 Iteration
    x1, y1 = run_complexity_analysis(A, rvd.deterministic_svd, ranks_to_test, x_metric='time', max_iter=1)
    # Run 2: 25 Iterations
    x2, y2 = run_complexity_analysis(A, rvd.deterministic_svd, ranks_to_test, x_metric='time', max_iter=25)
    
    plot_results(
        [(x1, y1, "Det. SVD (1 Iter)"), (x2, y2, "Det. SVD (25 Iter)")],
        title=f"Deterministic SVD: Error vs. Time\n({MATRIX_ROWS}x{MATRIX_COLS} Matrix)",
        xlabel="Time (seconds)"
    )

    # 4. Experiment: Randomized SVD (Error vs Rank)
    # Comparing different oversampling (p) and power iterations (q)
    print("\n--- Experiment: Randomized SVD Parameter Tuning ---")
    ranks_rand = range(1, 300, 1)
    
    # Config A: q=0, p=0
    xa, ya = run_complexity_analysis(A, rvd.rand_svd, ranks_rand, x_metric='rank', n_iter=0, oversample=0)
    # Config B: q=2, p=5
    xb, yb = run_complexity_analysis(A, rvd.rand_svd, ranks_rand, x_metric='rank', n_iter=2, oversample=5)
    
    plot_results(
        [(xa, ya, "Rand SVD (q=0, p=0)"), (xb, yb, "Rand SVD (q=2, p=5)")],
        title="Randomized SVD: Error Convergence by Rank",
        xlabel="Rank (k)"
    )
    """
    # 5. Experiment: Time Comparison (Deterministic vs Randomized) by Rank
    print("\n--- Experiment: Time Comparison (Deterministic vs Randomized) ---")
    time_ranks = range(1, 510, 20)
    det_times, rand_times = run_time_comparison_by_rank(
        A,
        time_ranks,
        det_kwargs={"max_iter": 25},
        rand_kwargs={"n_iter": 2, "oversample": 5},
    )

    plot_results(
        [
            (list(time_ranks), det_times, "Deterministic SVD"),
            (list(time_ranks), rand_times, "Randomized SVD"),
        ],
        title="SVD Laufzeitvergleich nach Rang",
        xlabel="Rank (k)",
        ylabel="Zeit (Sekunden)",
    )

if __name__ == "__main__":
    run_full_benchmark()
