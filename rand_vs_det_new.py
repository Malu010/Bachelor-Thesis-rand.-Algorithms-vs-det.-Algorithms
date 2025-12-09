import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
#------------- Randomized SVD -------------
def rand_svd (A, n_components, n_iter = 5, oversample=10):
    k_p = n_components + oversample # Etwas mehr Dimensionen für bessere Approximation
    rand_matrix = np.random.randn(A.shape[1], k_p) # Zufällige Normalverteilte Matrix mit weniger Spalten als die Originalmatrix
    y = A @ rand_matrix # Projektion der Originalmatrix auf die Zufallsmatrix (Skizzierung)
    for _ in range(n_iter): # Power-Iteration: bei jeder Iteration wird die Skizzierung verbessert und Rauschen reduiziert
        y = A.T @ y
        
        # WICHTIG: Sofort orthogonalisieren!
        # Das verhindert, dass alle Vektoren in die gleiche Richtung kippen.
        y, _ = np.linalg.qr(y, mode='reduced') 

        # Projektion zurück in den Spaltenraum
        y = A @ y
        
        # WICHTIG: Wieder orthogonalisieren!
        y, _ = np.linalg.qr(y, mode='reduced')
    q, r = np.linalg.qr(y, "reduced") # QR-Zerlegung der skizzierten Matrix
    b = q.T @ A # Reduzierte Matrix durch Projektion der Originalmatrix auf den Orthonormalbasisraum Q (weniger Zeilen)
    u, s, vT = np.linalg.svd(b, full_matrices=False) # SVD der reduzierten Matrix
    u = u[:, :n_components]
    s = s[: n_components]
    vt = vT[:n_components, :]
    u_final = q @ u # Endgültige linke Singulärvektoren
    return u_final, s, vt

#--------------- Deterministic Power Iteration SVD #########
def deterministic_svd(A, n_components, max_iter=100, tol=1e-5):
    m, n = A.shape
    
    #Eine Matrix die immer gleich ist aber deren Werte linear zwischen -1 und 1 liegen, um keine Richtung zu bevorzugen 
    # --> Startunterraum ist immer gleich --> deterministisch
    V = np.linspace(-1, 1, num=n*n_components).reshape(n, n_components)
    
    # Orthogonalisierung des Startblocks mit QR
    V, _ = np.linalg.qr(V)
    
    # --- Power Iteration --> Idee: Matrix wird wiederholt auf einen Vektor angewendet --> Vektor dreht sich in dominante Richtung ---
    for i in range(max_iter):
        V_old = V
        
        # Projektion in den Spaltenraum
        U = A @ V
        # QR damit Vektoren in U nicht alle in die gleiche Richtung kippen (Kollabieren?) -> k-stärkste Vektoren statt nur einen
        U, _ = np.linalg.qr(U, mode='reduced')
        
        # Projektion in den Zeilenraum 
        V = A.T @ U
        V, _ = np.linalg.qr(V, mode='reduced')
        
        # Einfacher Konvergenz-Check (Winkel zwischen Unterräumen)
        # Wir nutzen die Projektion: Wenn V und V_old gleich sind, ist die Norm ~ k
        if i > 0: # Erst ab zweiter Iteration prüfen
            #  Frobenius Norm der Differenz
            # abs wegen Vorzeichen (unterschiedliche Richtungen sind egal, nur Winkel zählt)
            diff = np.linalg.norm(np.abs(V.T @ V_old) - np.eye(n_components))
            if diff < tol:
                break
    
    # --- 3. Rayleigh-Ritz Schritt (Finale Werte extrahieren) ---
    # projiziere A auf die kleinen Dimensionen (k x k)
    B = U.T @ A @ V
    
    # Kleine SVD (deterministisch via numpy)
    U_small, s, Vt_small = np.linalg.svd(B)
    
    # Rückprojektion auf volle Größe
    U_final = U @ U_small
    VT_final = Vt_small @ V.T
    
    return U_final, s, VT_final
