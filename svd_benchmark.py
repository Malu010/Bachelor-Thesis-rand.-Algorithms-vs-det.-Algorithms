import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import rand_vs_det_new as rvd  
# k muss = lenght sein
# k ist die Anzahl der dominanten Singulärwerte
# length ist die Gesamtanzahl der Singulärwerte
def generate_large_matrix(m, n, k=200, length=200):
    # Orthogonale Matrizen (randomisiert)
    U = np.linalg.qr(np.random.randn(m, k))[0]
    V = np.linalg.qr(np.random.randn(n, k))[0]
    
    # Schnell abfallende Singulärwerte von 1e4 bis 1e-3
    sing_vals = np.geomspace(1e4, 1e-3, num=length) 
    
    # Low-Rank-Matrix A = U * Sigma * V^T
    Sigma = np.diag(sing_vals)
    A = U @ Sigma @ V.T
    
    return A  

#-- Rekonstruktionsfehler ---
def rel_error_rand(A, u_rand, s_rand, vt_rand):
        norm_A = np.linalg.norm(A, 'fro')
        A_reconstructed_rand = u_rand @ np.diag(s_rand) @ vt_rand
        norm_error = np.linalg.norm(A - A_reconstructed_rand, 'fro')  # ||A - A_k||_F
        relative_error = (norm_error / norm_A)*100
        return relative_error  
      
def benchmark():
    rows, cols = 1000, 1000 
    n_comp =  300            
    n_comp2 = 15
    n_comp3 = 1
    # Matrix mit schnell abfallenden Singulärwerten generieren
    np.random.seed(42)
    # k sind die dominanten Singulärwerte
    A = generate_large_matrix(rows, cols)
    
    #--------------------------
    print(f"--- Benchmark Start ---")
    print(f"Matrix Dimension: {rows}x{cols}")
    print(f"Gesuchte Komponenten (n_components): {n_comp}")
    
    #Benchmark rand. n_comp
    print(f"Running Randomized SVD comp= {n_comp}...")
    start = time.time()
    u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n_comp)
    end = time.time()
    relative_error_rand = rel_error_rand(A, u_rand, s_rand, vt_rand)
    
    #-------Fehlermessung randomized SVD
    
    
    # n_comp2
    print(f"Running Randomized SVD comp= {n_comp2}...")
    start2 = time.time()
    u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n_comp2)
    end2 = time.time()
    relative_error_rand2 = rel_error_rand(A, u_rand, s_rand, vt_rand)
    # n_comp3
    print(f"Running Randomized SVD comp= {n_comp3}...")
    start3 = time.time()
    u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n_comp3)
    end3 = time.time()
    relative_error_rand3 = rel_error_rand(A, u_rand, s_rand, vt_rand)
    
      
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
    print(f"Randomized SVD (n_components={n_comp}) took {end - start:.2f} seconds with reconstruction error {relative_error_rand:.2f}%")
    print(f"Randomized SVD (n_components={n_comp2}) took {end2 - start2:.2f} seconds with reconstruction error {relative_error_rand2:.2f}%")
    print(f"Randomized SVD (n_components={n_comp3}) took {end3 - start3:.2f} seconds with reconstruction error {relative_error_rand3:.2f}%")
    #-------Fehlermessung deterministic SVD
    def rel_error_det():
        norm_A = np.linalg.norm(A, 'fro')
        A_reconstructed_det = u_det @ np.diag(s_det) @ vt_det
        norm_error_det = np.linalg.norm(A - A_reconstructed_det, 'fro')
        relative_error_det = (norm_error_det / norm_A)*100
        return relative_error_det
    print(f"Deterministic SVD (n_components={cols}) took {end_det - start_det:.2f} seconds with reconstruction error {rel_error_det():.2f}%")
    
    
    #----- Entwicklung der Genauigkeit bei verschiedenen n_components mit Plots-----
    n_components_results = []
    components_amount = []
    """"
    for n in range(1, 1001, 50):
        u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n)
        relative_error_rand = rel_error_rand(A, u_rand, s_rand, vt_rand)
        n_components_results.append((relative_error_rand))
        print(f"n_components = {n}, Rel. Fehler = {relative_error_rand:2f}%")
        components_amount.append(n)
        if relative_error_rand < 1:
            break
        
    n_components_results_scaled = [error / 100 for error in n_components_results] 
    plt.figure(figsize=(8, 6))
    plt.plot(components_amount, n_components_results_scaled)
    plt.ylabel('Rekonstruktionsfehler in Decimal')
    plt.xlabel('Anzahl der Komponenten (n_components)')
    plt.title(f'Rekonstruktionsfehler vs. Anzahl der Komponenten bei Randomized SVD \n ({rows}x{cols} Matrix) k, length=200')
    plt.grid(True)
    plt.show()"""
    
    time_list = []
    for n in range(1, 1001, 50):
        start = time.time()
        u_rand, s_rand, vt_rand = rvd.rand_svd(A, n_components=n)
        end = time.time()
        time_taken = end - start
        relative_error_rand = rel_error_rand(A, u_rand, s_rand, vt_rand)
        n_components_results.append((relative_error_rand))
        print(f"n_components = {n}, Dauer = {time_taken:.2f} seconds")
        time_list.append(time_taken)
        components_amount.append(n)
        if relative_error_rand < 1.0:
            break
    n_components_results_scaled = [error / 100 for error in n_components_results] 
    plt.figure(figsize=(8, 6))
    plt.plot(time_list, n_components_results_scaled)
    plt.ylabel('Rekonstruktionsfehler in Decimal')
    plt.xlabel('Zeit in Sekunden')
    plt.title(f'Rekonstruktionsfehler vs. Zeit in Sekunden bei Randomized SVD \n ({rows}x{cols} Matrix) k, length=200')
    plt.grid(True)
    plt.show()
    

        
        

if __name__ == "__main__":
    benchmark()  # --  