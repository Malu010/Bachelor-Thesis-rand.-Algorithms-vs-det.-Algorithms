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
    'stable': {},   # n_iter = 10
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
plt.savefig('results/rsvd_eigenfaces_depth_comparison.png', dpi=300)
plt.show()

# --- ZUSATZ: Quantitativer Vergleich (Euklidische Distanz) ---
print("Berechne euklidische Distanzen...")

distances_stable = []
distances_unstable = []

# Matrizen für Seed 1 und Seed 2 extrahieren
vt_stable_A = results['stable'][seeds[0]]
vt_stable_B = results['stable'][seeds[1]]

vt_unstable_A = results['unstable'][seeds[0]]
vt_unstable_B = results['unstable'][seeds[1]]

for i in range(n_comp_calc):
    # 1. Stabile Version (n_iter = 10)
    vA_s = vt_stable_A[i]
    vB_s = vt_stable_B[i]
    # Vorzeichen-Ambiguität abfangen: min(||vA - vB||, ||vA + (-vB)||)
    dist_s = min(np.linalg.norm(vA_s - vB_s), np.linalg.norm(vA_s + vB_s))
    distances_stable.append(dist_s)
    
    # 2. Instabile Version (n_iter = 0)
    vA_u = vt_unstable_A[i]
    vB_u = vt_unstable_B[i]
    dist_u = min(np.linalg.norm(vA_u - vB_u), np.linalg.norm(vA_u + vB_u))
    distances_unstable.append(dist_u)

# Plotting des Graphen
fig_dist, ax_dist = plt.subplots(figsize=(10, 6))

x_axis = np.arange(1, n_comp_calc + 1) # x-Achse: 1 bis 60

ax_dist.plot(x_axis, distances_unstable, label='Ohne Power Iterations ($q=0$)', 
             color='red', marker='o', markersize=4, linestyle='-', alpha=0.7)
ax_dist.plot(x_axis, distances_stable, label='Mit Power Iterations ($q=10$)', 
             color='blue', marker='s', markersize=4, linestyle='-', alpha=0.8)

# Achsenbeschriftung und Design
ax_dist.set_title("Euklidische Distanz der Eigenfaces zwischen zwei Random-Seeds", fontsize=14, pad=15)
ax_dist.set_xlabel("Index des Eigenfaces ($k$)", fontsize=12)
ax_dist.set_ylabel("Minimale Euklidische Distanz", fontsize=12)


# ax_dist.set_yscale('log')

ax_dist.grid(True, linestyle='--', alpha=0.6)
ax_dist.legend(fontsize=12, loc='upper left')

# Markiere die Indizes der geplotteten Eigenfaces mit vertikalen Linien und Text
for idx in [0, 9, 49]:
    ax_dist.axvline(x=idx+1, color='gray', linestyle=':', alpha=0.5)
    ax_dist.text(idx+1.5, ax_dist.get_ylim()[1]*0.8, f'{idx+1}. EF', color='gray', fontsize=10)

plt.tight_layout()
plt.savefig('results/rsvd_quantitative_distance.png', dpi=300)
plt.show()


