import time

import matplotlib.pyplot as plt
import numpy as np

import rand_vs_det_new as rvd


def generate_large_matrix(m: int, n: int, length: int = 200) -> np.ndarray:
    """Create a matrix with rapidly decaying singular values."""
    u = np.linalg.qr(np.random.randn(m, length))[0]
    v = np.linalg.qr(np.random.randn(n, length))[0]
    sing_vals = np.geomspace(1e4, 1e-3, num=length)
    sigma = np.diag(sing_vals)
    return u @ sigma @ v.T


def compute_relative_error(A: np.ndarray, u: np.ndarray, s: np.ndarray, vt: np.ndarray) -> float:
    """Return relative Frobenius reconstruction error in percent."""
    norm_A = np.linalg.norm(A, "fro")
    reconstructed = u @ np.diag(s) @ vt
    norm_error = np.linalg.norm(A - reconstructed, "fro")
    return float((norm_error / norm_A) * 100)


def run_randomized_trials(A: np.ndarray, components: list[int]) -> list[tuple[int, float, float]]:
    """Run randomized SVD for multiple component counts and collect timings and errors."""
    results = []
    for n_comp in components:
        print(f"Running Randomized SVD comp={n_comp}...")
        start = time.time()
        u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n_comp)
        duration = time.time() - start
        rel_error = compute_relative_error(A, u_rand, s_rand, vt_rand)
        print(
            f"Randomized SVD (n_components={n_comp}) took {duration:.2f} seconds "
            f"with reconstruction error {rel_error:.2f}%"
        )
        results.append((n_comp, duration, rel_error))
    return results


def run_deterministic_trial(A: np.ndarray, n_components: int) -> tuple[float, float]:
    """Run deterministic SVD once and return duration and reconstruction error."""
    print("Running Deterministic SVD comp= full...")
    start = time.time()
    u_det, s_det, vt_det = rvd.deterministic_svd(A, n_components=n_components)
    duration = time.time() - start
    rel_error = compute_relative_error(A, u_det, s_det, vt_det)
    print(
        f"Deterministic SVD (n_components={n_components}) took {duration:.2f} seconds "
        f"with reconstruction error {rel_error:.2f}%"
    )
    return duration, rel_error


def det_exp_comp_vs_error(
    A: np.ndarray, start: int, end: int, step: int, n_iter: int
) -> tuple[list[int], list[float]]:
    """Sweep deterministic SVD over component counts until error drops below 1%."""
    components_amount: list[int] = []
    n_components_results: list[float] = []
    for n in range(start, end, step):
        u_det, s_det, vt_det = rvd.deterministic_svd(A, n_components=n, max_iter=n_iter)
        relative_error_det = compute_relative_error(A, u_det, s_det, vt_det)
        n_components_results.append(relative_error_det)
        components_amount.append(n)
        print(f"n_components = {n}, Relative error = {relative_error_det:2f}%")
        if relative_error_det < 1:
            break
    return components_amount, n_components_results


def plot_deterministic_error_progression(rows: int, cols: int, A: np.ndarray) -> None:
    """Plot reconstruction error vs. rank for multiple deterministic iteration counts."""
    x1, y1 = det_exp_comp_vs_error(A, 1, 1100, 1, 1)
    x2, y2 = det_exp_comp_vs_error(A, 1, 1100, 1, 25)
    x3, y3 = det_exp_comp_vs_error(A, 1, 1100, 1, 100)

    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label="1 Iteration")
    plt.plot(x2, y2, label="25 Iterationen")
    plt.plot(x3, y3, label="100 Iterationen")

    plt.ylabel("Reconstruction error (decimal)")
    plt.xlabel("Rank of approximation k")
    plt.title(
        f"Reconstruction error vs. approximation rank for deterministic SVD\n "
        f"({rows}x{cols} matrix with exponential decay)"
    )
    plt.grid(True)
    plt.legend()
    plt.show()


def benchmark() -> None:
    rows, cols = 1000, 1000
    np.random.seed(42)

    print("--- Benchmark Start ---")
    print(f"Matrix Dimension: {rows}x{cols}")

    A = generate_large_matrix(rows, cols)
    n_comp_primary = [300, 15, 1]

    run_randomized_trials(A, n_comp_primary)
    run_deterministic_trial(A, n_components=300)
    plot_deterministic_error_progression(rows, cols, A)


if __name__ == "__main__":
    benchmark()