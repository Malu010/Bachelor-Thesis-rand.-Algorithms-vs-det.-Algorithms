import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from rand_vs_det_new import rand_svd


# Daten
print("Lade Daten...")
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data
# Zuerst Mean Face berechnen und von allen Bildern abziehen (zentrieren), damit die SVD die Varianz um den Mittelwert herum erfasst
mean_face = X.mean(axis=0)
X_centered = X - mean_face

# Parameter
indices_to_plot = [0, 9, 49] # 1., 10. und 50. Eigenface
seeds = [42, 999]            # Zwei verschiedene Seeds zum Vergleich
n_comp_calc = 60             # k 

# results['method_name'][seed] = vt_matrix
results = {
    'stable': {},   # n_iter = 2
    'unstable': {}  # n_iter = 0
}

print("Berechne rSVDs...")

# Berechnung mit Power Iterations (stabil)
for seed in seeds:
    np.random.seed(seed)
    _, _, vt = rand_svd(X_centered, n_components=n_comp_calc, n_iter=10, oversample=10)
    results['stable'][seed] = vt

# Berechnung ohne Power Iterations (instabil)
for seed in seeds:
    np.random.seed(seed)
    _, _, vt = rand_svd(X_centered, n_components=n_comp_calc, n_iter=0, oversample=10)
    results['unstable'][seed] = vt

# Plotting
fig, axes = plt.subplots(len(indices_to_plot), 4, figsize=(16, 12))

# Spalten-Titel setzen
cols = [f"Stabil (Seed {seeds[0]})", f"Stabil (Seed {seeds[1]})", 
        f"Instabil (Seed {seeds[0]})", f"Instabil (Seed {seeds[1]})"]

for ax, col in zip(axes[0], cols):
    ax.set_title(col, fontsize=12, fontweight='bold', pad=15)

# Zeilen iterieren (1., 10., 50. Eigenface)
for row_idx, eigen_idx in enumerate(indices_to_plot):
    
    # Beschriftung links (Welches Eigenface?)
    axes[row_idx, 0].set_ylabel(f"{eigen_idx + 1}. Eigenface", fontsize=14, rotation=90, labelpad=10)
    
    # --- Linke Seite: Stabil  ---
    # Bild 1 (Seed A)
    img = results['stable'][seeds[0]][eigen_idx].reshape(64, 64)
    axes[row_idx, 0].imshow(img, cmap='gray')
    
    # Bild 2 (Seed B)
    img = results['stable'][seeds[1]][eigen_idx].reshape(64, 64)
    axes[row_idx, 1].imshow(img, cmap='gray')
    
    # --- Rechte Seite: Instabil  ---
    # Bild 3 (Seed A)
    img = results['unstable'][seeds[0]][eigen_idx].reshape(64, 64)
    axes[row_idx, 2].imshow(img, cmap='gray')
    
    # Bild 4 (Seed B)
    img = results['unstable'][seeds[1]][eigen_idx].reshape(64, 64)
    axes[row_idx, 3].imshow(img, cmap='gray')

    # Achsen für alle entfernen
    for ax in axes[row_idx]:
        ax.set_xticks([])
        ax.set_yticks([])

# Ober-Überschriften für die Gruppen
fig.text(0.3, 0.95, "Mit Power Iterations", 
         ha='center', fontsize=14, fontweight='bold')
fig.text(0.72, 0.95, "Ohne Power Iterations", 
         ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.88, left=0.05) # Platz für Header lassen
plt.savefig('rsvd_eigenfaces_depth_comparison.png', dpi=300)
plt.show()


