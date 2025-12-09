import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import rand_vs_det_new as rvd  

def generate_large_matrix(m, n, k=50, length=50):
    # Orthogonale Matrizen (randomisiert)
    U = np.linalg.qr(np.random.randn(m, k))[0]
    V = np.linalg.qr(np.random.randn(n, k))[0]
    
    # Schnell abfallende Singulärwerte von 1e4 bis 1e-3
    sing_vals = np.geomspace(1e4, 1e-3, num=length) 
    
    # Low-Rank-Matrix A = U * Sigma * V^T
    Sigma = np.diag(sing_vals)
    A = U @ Sigma @ V.T
    
    return A  
  
def benchmark():
    rows, cols = 1500, 1500 
    n_comp = 10             
    n_comp2 = 5
    n_comp3 = 1
    
    print(f"--- Benchmark Start ---")
    print(f"Matrix Dimension: {rows}x{cols}")
    print(f"Gesuchte Komponenten (n_components): {n_comp}")
    
    # Matrix mit schnell abfallenden Singulärwerten generieren
    np.random.seed(42)
    A = generate_large_matrix(rows, cols, k=50, length=50)
    print(f"Running Randomized SVD comp= {n_comp}...")
    #Benchmark rand.
    start = time.time()
    u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n_comp)
    end = time.time()
    
    #Benchmark det.
    print(f"Running Deterministic SVD comp= full...")
    start_det = time.time()
    u_det, s_det, vt_det = rvd.deterministic_svd(A, n_components=cols)
    end_det = time.time()
    

    # Zufällige Matrix erstellen
    #np.random.seed(42)
    #A = np.random.randint(low=-100, high=100, size=(rows, cols))
    
    #--- Randomized SVD Benchmark n = 150---
    
    #print(f"Running Randomized SVD comp = {n_comp}...")
    #start = time.time()
    #u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n_comp)
    #end = time.time()
    
    
    
    
    
    #-------Fehlermessung randomized SVD
    norm_A = np.linalg.norm(A, 'fro')
    A_reconstructed_rand = u_rand @ np.diag(s_rand) @ vt_rand
    norm_error = np.linalg.norm(A - A_reconstructed_rand, 'fro')  # ||A - A_k||_F
    relative_error = (norm_error / norm_A)*100
    print(f"Randomized SVD (n_components={n_comp}) took {end - start:.2f} seconds with reconstruction error {relative_error:.2f}%")
    
    
    #-------Fehlermessung deterministic SVD
    A_reconstructed_det = u_det @ np.diag(s_det) @ vt_det
    norm_error_det = np.linalg.norm(A - A_reconstructed_det, 'fro')
    relative_error_det = (norm_error_det / norm_A)*100
    print(f"Deterministic SVD (n_components={cols}) took {end_det - start_det:.2f} seconds with reconstruction error {relative_error_det:.2f}%")
    
    
    #----- Entwicklung der Genauigkeit bei verschiedenen n_components mit Plots-----
"""    n_components_results = []
    components_amount = []
    for n in range(1, 1502, 150):
        u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n)
        A_reconstructed_rand = u_rand @ np.diag(s_rand) @ vt_rand
        norm_error = np.linalg.norm(A - A_reconstructed_rand, 'fro')  # ||A - A_k||_F
        relative_error = round(norm_error / norm_A, 4) * 100  # In Prozent
        n_components_results.append((relative_error))
        print(f"n_components={n}, Rekonstruktionsfehler={relative_error:.2f}%") 
        components_amount.append(n)
        
        n += 10  
    plt.figure(figsize=(8, 6))
    plt.plot(components_amount, n_components_results, 'o-', linewidth=2)
    plt.semilogy()
    plt.ylabel('Rel. Fehler × 100 %')
    plt.title('Randomized SVD: Rekonstruktionsfehler bei n_components')
    plt.show()"""

        
        

if __name__ == "__main__":
    benchmark()  # --  