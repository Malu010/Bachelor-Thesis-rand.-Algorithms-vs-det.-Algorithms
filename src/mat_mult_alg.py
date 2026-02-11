import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

def matmul_loops(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    
    assert cols_A == rows_B, f"Dimension mismatch: {cols_A} != {rows_B}"
    C = np.zeros((rows_A, cols_B))
    for i in range(rows_A):
        for j in range(cols_B):
            total = 0.0
            for k in range(cols_A):
                total += A[i, k] * B[k, j]
            C[i, j] = total
            
    return C
# randomisiert mit Sketching
def randomized_range_matmul(A, B, rank_target=50, oversample=10):
    k = rank_target + oversample
    m, n = A.shape
    
    # Zufallsmatrix Omega (n x k)
    Omega = np.random.randn(n, k)
    
    Y = matmul_loops(A, Omega)
    
    #QR Zerlegung. Nach Halko A = Q Q^T A
    Q, _ = np.linalg.qr(Y)
    # Q ist orthonormal (m x k)
    
    # Small_A = Q.T @ A
    Small_A = matmul_loops(Q.T, A) # Q.T ist (k x m), A ist (m x n)
    
    # Small_C = Small_A @ B
    Small_C = matmul_loops(Small_A, B) # (k x n) * (n x p)
    
    # C_approx = Q @ Small_C
    C_approx = matmul_loops(Q, Small_C) # (m x k) * (k x p)
    
    return C_approx

def naive_randomized_matmul(A, B, sketch_dim=50):
    # Zufallsmatrix S (n x sketch_dim)
    n = A.shape[1]
    S = np.random.randn(n, sketch_dim)
    
    # Skizzierte Matrizen
    A_sketch = matmul_loops(A, S)  # (m x n) * (n x sketch_dim) = (m x sketch_dim)
    B_sketch = matmul_loops(S.T, B)  # (sketch_dim x n) * (n x p) = (sketch_dim x p)
    
    # Approximierte Matrixmultiplikation
    C_approx = matmul_loops(A_sketch, B_sketch)  # (m x sketch_dim) * (sketch_dim x p) = (m x p)
    
    return C_approx/sketch_dim


# deterministisch mit Schleife
def det_matmult(A, B):
    m, nA = A.shape
    nB, k = B.shape
    assert nA == nB
    C = matmul_loops(A, B)
    return C


# Fehlerrate

"""def rel_mm_error(A, B, C_rand):
    C_exact = A @ B
    num = np.linalg.norm(C_exact - C_rand, 'fro')
    den = np.linalg.norm(C_exact, 'fro')
    return 100 * num / den
"""
# Benchmark

def bench_plot_comparison():
    sizes = range(0, 100, 1)
    error_naive = []
    error_stable = []
    
    print(f"{'Size':<10} | {'Naive Error':<15} | {'Stable Error (Range Finder)':<25}")
    print("-" * 60)

    for size in sizes:
        # Matrix mit Rang = true_rank
        true_rank = 50
        A = np.random.randn(size, true_rank) @ np.random.randn(true_rank, size)
        B = np.random.randn(size, true_rank) @ np.random.randn(true_rank, size)
        
        sketch_dim = 100 
        # naive Methode
        #S = np.random.randn(size, sketch_dim)
        #C_naive = naive_randomized_matmul(A, B, sketch_dim)
        
        # Halko Methode
        C_stable = randomized_range_matmul(A, B, rank_target=sketch_dim)
        
        # Exakt
        C_exact = A @ B
        
        # Fehler berechnen
        #err_n = 100 * np.linalg.norm(C_exact - C_naive, 'fro') / np.linalg.norm(C_exact, 'fro')
        err_s = 100 * np.linalg.norm(C_exact - C_stable, 'fro') / np.linalg.norm(C_exact, 'fro')
        
        #error_naive.append(err_n)
        error_stable.append(err_s)
        
        print(f"{size:<10} | {'N/A':<15} | {err_s:.10f}%")
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, error_naive, label='Naive Randomized Matmul', color='blue')
    plt.xlabel('Matrix Size')
    plt.ylabel('Relativer Fehler (%)')
    plt.title(f'Naive Matrixmultiplikation: Matrix Size vs. Relativer Fehler. Rang = {true_rank} ')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, error_stable, label=f'k = {sketch_dim}', color='green')
    plt.xlabel('Matrix Size')
    plt.ylabel('Relativer Fehler (%)')
    plt.title(f'Matmul mit QR: Matrix Size vs. Relativer Fehler. Rang = {true_rank} ')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()
    
def run_comprehensive_benchmark():
    # 1. Konfiguration
    # Wir testen verschiedene k-Werte (Sketch Dimensions)
    # k_values = [20, 40, 60] -> Wenig, Mittel, Hohe Genauigkeit
    k_values = [70, 100, 130] 
    true_rank = 70 # Die echte Datenstruktur ist klein
    
    # Größen testen (Schritte von 20 bis 160). 
    # Achtung: Bei Loops > 200 dauert es sehr lange!
    sizes = range(30, 500, 50) 
    
    # Speicher für Ergebnisse
    results = {
        'sizes': [],
        'det_time': [],
        'rand_times': {k: [] for k in k_values},
        'rand_errors': {k: [] for k in k_values}
    }

    print(f"{'Size':<5} | {'Det Time':<10} | {'k':<3} | {'Rand Time':<10} | {'Error %':<15} | {'Speedup':<10}")
    print("-" * 75)

    for size in sizes:
        results['sizes'].append(size)
        
        # Erstelle Low-Rank Matrizen (Rang 70)
        A = np.random.randn(size, true_rank) @ np.random.randn(true_rank, size)
        B = np.random.randn(size, true_rank) @ np.random.randn(true_rank, size)
        
        # 1. Deterministisch (Baseline)
        start = time.time()
        C_exact = det_matmult(A, B)
        det_dur = time.time() - start
        results['det_time'].append(det_dur)
        
        norm_exact = np.linalg.norm(C_exact, 'fro')
        
        # 2. Randomized für verschiedene k
        for k in k_values:
            start = time.time()
            # Wir nutzen die stabile QR Methode (Randomized Range Finder)
            C_approx = randomized_range_matmul(A, B, rank_target=k, oversample=5)
            rand_dur = time.time() - start
            
            # Fehler berechnen
            diff = np.linalg.norm(C_exact - C_approx, 'fro')
            rel_error = 100 * diff / norm_exact
            
            results['rand_times'][k].append(rand_dur)
            results['rand_errors'][k].append(rel_error)
            
            # Konsolenausgabe
            speedup = det_dur / rand_dur if rand_dur > 0 else 0
            print(f"{size:<5} | {det_dur:.4f}s    | {k:<3} | {rand_dur:.4f}s    | {rel_error:.6f}%    | {speedup:.2f}x")
        
        print("-" * 75)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Geschwindigkeit (Zeit vs. Größe)
    ax1.plot(sizes, results['det_time'], label='Deterministic (Exact)', color='black', linewidth=2, linestyle='--')
    
    colors = ['green', 'blue', 'purple']
    for i, k in enumerate(k_values):
        ax1.plot(sizes, results['rand_times'][k], label=f'Randomized (k={k})', color=colors[i])
    
    ax1.set_xlabel('Matrix Dimension (n x n)')
    ax1.set_ylabel('Rechenzeit (Sekunden)')
    ax1.set_title('Performance Vergleich')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Genauigkeit (Fehler vs. Größe)
    # Wir benutzen hier eine logarithmische Skala für den Fehler, da er sehr klein werden kann
    for i, k in enumerate(k_values):
        ax2.plot(sizes, results['rand_errors'][k], label=f'k={k}', color=colors[i])
        
    ax2.set_xlabel('Matrix Dimension (n x n)')
    ax2.set_ylabel('Relativer Fehler (%)')
    ax2.set_yscale('log') # Logarithmische Skala!
    ax2.set_title(f'Genauigkeit (Log-Skala) | True Rank={true_rank}')
    
    
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.5)

    plt.tight_layout()
    plt.show()    
if __name__ == "__main__":
    run_comprehensive_benchmark()