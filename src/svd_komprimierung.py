import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rand_vs_det_new as rvd

# ==========================================
# 1. QUANTISIERUNG UND KODIERUNGS-HELFER
# ==========================================

def quantize_and_threshold(matrix, epsilon=0.01, bits=8):
    """Wendet Thresholding und Quantisierung auf eine Matrix an."""
    max_abs_val = np.max(np.abs(matrix))
    if max_abs_val == 0:
        return np.zeros_like(matrix, dtype=np.int32), 1.0, 100.0
        
    threshold_value = epsilon * max_abs_val
    thresholded = np.where(np.abs(matrix) < threshold_value, 0.0, matrix)
    
    max_int = (2 ** (bits - 1)) - 1 
    scale_factor = max_int / max_abs_val
    quantized_matrix = np.round(thresholded * scale_factor).astype(np.int32)
    
    sparsity = np.mean(quantized_matrix == 0) * 100
    return quantized_matrix, scale_factor, sparsity

def dequantize(quantized_matrix, scale_factor):
    """Macht die Quantisierung für die Bildrekonstruktion wieder rückgängig."""
    return quantized_matrix.astype(np.float64) / scale_factor

def calculate_pure_huffman_bytes(quantized_matrix):
    """
    Methode 1: Nur Huffman (Shannon-Entropie auf rohen Daten).
    """
    flat = quantized_matrix.flatten()
    if len(flat) == 0: return 0
    _, counts = np.unique(flat, return_counts=True)
    probabilities = counts / len(flat)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return int(np.ceil((entropy * len(flat)) / 8.0))

def calculate_rle_variants(quantized_matrix):
    """
    Berechnet die Dateigröße für:
    1. Nur RLE (Lauflängenkodierung)
    2. RLE + Huffman (Entropie des RLE-Streams)
    """
    flat = quantized_matrix.flatten()
    if len(flat) == 0: return 0, 0
    
    # Lauflängen berechnen
    # Indizes finden, wo sich der Wert ändert
    changes = np.where(flat[:-1] != flat[1:])[0] + 1
    # Indizes splitten: Start(0), Änderungen, Ende
    split_indices = np.concatenate(([0], changes, [len(flat)]))
    
    run_lengths = np.diff(split_indices)
    values = flat[split_indices[:-1]]
    
    # --- Methode 2: Nur RLE ---
    # Protokoll: [Anzahl (1 Byte), Wert (1 Byte)]
    # Wenn eine Länge > 255 ist, braucht man mehrere Pakete.
    # Anzahl der Pakete für eine Länge L = ceil(L / 255)
    rle_chunks = np.ceil(run_lengths / 255.0)
    bytes_rle_only = int(np.sum(rle_chunks) * 2)
    
    # --- Methode 3: Huffman + RLE ---
    # Symbol-Stream: [L1, V1, L2, V2, ...]
    # Lange Runs werden gesplittet: 255, V, Rest, V
    stream = []
    for L, V in zip(run_lengths, values):
        while L > 255:
            stream.extend([255, V])
            L -= 255
        if L > 0:
            stream.extend([L, V])
            
    if not stream:
        bytes_rle_huff = 0
    else:
        # Entropie dieses Streams berechnen
        stream_arr = np.array(stream)
        _, counts = np.unique(stream_arr, return_counts=True)
        probs = counts / len(stream_arr)
        entropy = -np.sum(probs * np.log2(probs))
        bytes_rle_huff = int(np.ceil((entropy * len(stream_arr)) / 8.0))
        
    return bytes_rle_only, bytes_rle_huff

# ==========================================
# 2. HAUPTKOMPRESSIONS-FUNKTION
# ==========================================

def svd_compression(img, n_components, epsilon=0.01, bits=8):    
    """
    Komprimiert ein Bild und gibt Größen für ALLE 3 Methoden zurück.
    Rückgabe: (reconstructed_img, sizes_dict)
    """
    img_float = img.astype(np.float64)
    
    # Initialisierung der Zähler für die 3 Methoden
    sizes = {'pure_huff': 0, 'rle_only': 0, 'rle_huff': 0}
    
    # Hilfsfunktion zum Aufaddieren der Größen eines Kanals/Matrix
    def add_matrix_sizes(matrix):
        # 1. Huffman pur
        sizes['pure_huff'] += calculate_pure_huffman_bytes(matrix) + 4 # +4 für Scale Factor
        
        # 2. RLE Varianten
        rle_only, rle_huff = calculate_rle_variants(matrix)
        sizes['rle_only'] += rle_only + 4
        sizes['rle_huff'] += rle_huff + 4

    # Metadaten-Größe für Singulärwerte (S) ist für alle gleich
    s_bytes = n_components * 4 
    
    if len(img.shape) == 2:
        u, s, v = rvd.rand_svd(img_float, n_components)
        u_quant, u_scale, _ = quantize_and_threshold(u, epsilon, bits)
        v_quant, v_scale, _ = quantize_and_threshold(v, epsilon, bits)
        
        add_matrix_sizes(u_quant)
        add_matrix_sizes(v_quant)
        
        reconstructed = dequantize(u_quant, u_scale) @ np.diag(s) @ dequantize(v_quant, v_scale)
        
    elif len(img.shape) == 3:
        reconstructed = np.zeros_like(img_float)
        for channel in range(img.shape[2]):
            u, s, v = rvd.rand_svd(img_float[:, :, channel], n_components)
            u_quant, u_scale, _ = quantize_and_threshold(u, epsilon, bits)
            v_quant, v_scale, _ = quantize_and_threshold(v, epsilon, bits)
            
            add_matrix_sizes(u_quant)
            add_matrix_sizes(v_quant)
            
            reconstructed[:, :, channel] = dequantize(u_quant, u_scale) @ np.diag(s) @ dequantize(v_quant, v_scale)
            
        # Singulärwerte fallen pro Kanal an
        s_bytes *= 3
            
    else:
        raise ValueError("Bild muss 2D oder 3D sein.")
    
    # S-Vektor zu allen Methoden addieren
    for key in sizes:
        sizes[key] += s_bytes

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed, sizes

# ==========================================
# 3. PLOTTING UND EVALUATION
# ==========================================

def plot_image_compressions(original_img, k_values, epsilon=0.01):
    n_plots = 1 + len(k_values)
    cols = 3
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    cmap = 'gray' if len(original_img.shape) == 2 else None
    
    m, n = original_img.shape[:2]
    c = 1 if len(original_img.shape) == 2 else original_img.shape[2]
    original_kb = (m * n * c) / 1024
    
    axes[0].imshow(original_img, cmap=cmap)
    axes[0].set_title(f"Originalbild\nUnkomprimiert: {original_kb:.1f} KB")
    axes[0].axis('off')
    
    print(f"Berechne {len(k_values)} Bilder für den visuellen Vergleich...")
    for i, k in enumerate(k_values):
        comp_img, sizes = svd_compression(original_img, k, epsilon=epsilon)
        comp_kb = sizes['pure_huff'] / 1024
        
        axes[i+1].imshow(comp_img, cmap=cmap)
        axes[i+1].set_title(f"Komprimiert (k = {k})\nGröße (Huff): {comp_kb:.1f} KB")
        axes[i+1].axis('off')
        
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_size_vs_mse(original_img, epsilon=0.01):
    m, n = original_img.shape[:2]
    c = 1 if len(original_img.shape) == 2 else original_img.shape[2]
    original_bytes = m * n * c
    
    k_values = list(range(10, 310, 20)) 
    mses = []
    sizes_kb = []
    
    print("Berechne Daten für den Trade-off Graphen...")
    for k in k_values:
        comp_img, sizes = svd_compression(original_img, k, epsilon=epsilon)
        mse = np.mean((original_img.astype(np.float64) - comp_img.astype(np.float64)) ** 2)
        mses.append(mse)
        sizes_kb.append(sizes['pure_huff'] / 1024)
        
    plt.figure(figsize=(10, 6))
    plt.plot(mses, sizes_kb, marker='o', linestyle='-', color='b', linewidth=2)
    plt.axhline(y=original_bytes/1024, color='r', linestyle='--', label=f'Original ({original_bytes/1024:.1f} KB)')
    plt.xlabel("Fehler: MSE", fontsize=12)
    plt.ylabel("Dateigröße (KB) [Pure Huffman]", fontsize=12)
    plt.title(f"Kompressionsleistung (Quantisierung $\epsilon={epsilon}$)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_compression_methods_comparison(original_img, epsilon=0.01):
    """
    Vergleicht die Dateneinsparung der 3 Methoden
    (Huffman vs. Huffman+RLE vs. RLE Only) über verschiedene k-Werte.
    """
    m, n = original_img.shape[:2]
    c = 1 if len(original_img.shape) == 2 else original_img.shape[2]
    original_kb = (m * n * c) / 1024
    
    k_values = list(range(10, 600, 30))
    
    # Listen für die 3 Methoden
    s_pure = []
    s_rle = []
    s_rle_huff = []
    
    print("Berechne Daten für den Methoden-Vergleich...")
    for k in k_values:
        _, sizes = svd_compression(original_img, k, epsilon=epsilon)
        s_pure.append(sizes['pure_huff'] / 1024)
        s_rle.append(sizes['rle_only'] / 1024)
        s_rle_huff.append(sizes['rle_huff'] / 1024)
        
    plt.figure(figsize=(10, 6))
    
    # Drei Linien plotten
    plt.plot(k_values, s_pure, marker='o', label='Huffman', linewidth=2)
    plt.plot(k_values, s_rle_huff, marker='s', label='Huffman + RLE', linewidth=2, linestyle='--')
    plt.plot(k_values, s_rle, marker='^', label='RLE', linewidth=2, linestyle=':')
    
    plt.axhline(y=original_kb, color='r', linestyle='-', alpha=0.5, label='Originalbild')
    
    plt.xlabel("Rang k (Anzahl Singulärwerte)", fontsize=12)
    plt.ylabel("Dateigröße (KB)", fontsize=12)
    plt.title(f"Vergleich der Kodierungsmethoden ($\epsilon={epsilon}$)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. MAIN BLOCK
# ==========================================
if __name__ == "__main__":
    img_path = r"img\Kirche.tiff"
    
    try:
        pil_img = Image.open(img_path)
        pil_img.thumbnail((800, 800)) 
        img = np.array(pil_img)
        
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
            
        test_epsilon = 0.05
        
        # Visueller Vergleich
        plot_image_compressions(img, [10, 50, 150, 200, 350], epsilon=test_epsilon)
        
        # Standard Trade-off
        plot_size_vs_mse(img, epsilon=test_epsilon)
        
        # Comprimierungsvergleich der Methoden
        plot_compression_methods_comparison(img, epsilon=test_epsilon)
        
    except FileNotFoundError:
        print(f"Datei nicht gefunden: {img_path}")