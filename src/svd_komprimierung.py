import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rand_vs_det_new as rvd  # Deine eigene rSVD-Implementierung

# ==========================================
# 1. QUANTISIERUNG UND LAUFLÄNGENKODIERUNG
# ==========================================

def quantize_and_threshold(matrix, epsilon=0.01, bits=8):
    """Wendet Thresholding und Quantisierung auf eine Matrix an."""
    
    # Den absolut größten Wert der Matrix finden
    max_abs_val = np.max(np.abs(matrix))
    if max_abs_val == 0:
        return np.zeros_like(matrix, dtype=np.int32), 1.0, 100.0
        
    # Treshold relativ zum größten Wert: Werte, die kleiner als epsilon * max_abs_val sind, werden auf 0 gesetzt
    threshold_value = epsilon * max_abs_val
    
    # 1. Thresholding: Werte innerhalb der relativen Totzone auf 0 setzen
    thresholded = np.where(np.abs(matrix) < threshold_value, 0.0, matrix)
    
    # 2. Quantisierung auf Integer
    max_int = (2 ** (bits - 1)) - 1 
    scale_factor = max_int / max_abs_val
    quantized_matrix = np.round(thresholded * scale_factor).astype(np.int32)
    
    sparsity = np.mean(quantized_matrix == 0) * 100
    return quantized_matrix, scale_factor, sparsity

def dequantize(quantized_matrix, scale_factor):
    """Macht die Quantisierung für die Bildrekonstruktion wieder rückgängig."""
    return quantized_matrix.astype(np.float64) / scale_factor

def calculate_rle_huffman_bytes(quantized_matrix):
    """
    Simuliert Hybrid-RLE (PackBits-Logik) gefolgt von perfekter Entropiekodierung.
    Wendet RLE erst ab 3 gleichen Werten an. Einzelgänger werden als Literals gebündelt.
    """
    flat = quantized_matrix.flatten()
    if len(flat) == 0:
        return 0
        
    # 1. Finde die Lauflängen und die dazugehörigen Werte
    changes = np.where(flat[:-1] != flat[1:])[0] + 1
    split_indices = np.concatenate(([0], changes, [len(flat)]))
    run_lengths = np.diff(split_indices)
    values = flat[split_indices[:-1]]
    
    stream = []
    literal_buffer = []
    
    # Hilfsfunktion, um angesammelte Einzelgänger in den Stream zu schreiben
    def flush_literals():
        nonlocal literal_buffer, stream
        while len(literal_buffer) > 0:
            # Ein PackBits-Header kann maximal 128 Literals auf einmal ankündigen
            chunk = literal_buffer[:128]
            literal_buffer = literal_buffer[128:]
            
            # Header für Literals: Länge - 1 (also Werte von 0 bis 127)
            stream.append(len(chunk) - 1)
            # Danach die echten Werte roh anhängen
            stream.extend(chunk)

    # 2. Den intelligenten Datenstrom aufbauen
    for L, val in zip(run_lengths, values):
        # Bedingung: RLE lohnt sich erst ab 3 gleichen Werten!
        if L >= 3:
            flush_literals()
            
            # Jetzt den Run verarbeiten (kann in PackBits max. 128 Zeichen lang sein)
            while L > 0:
                chunk_size = min(L, 128)
                # Header für Runs: Wir nutzen hohe Zahlen (129 bis 256), 
                # um sie von den Literal-Headern (0 bis 127) zu unterscheiden.
                header = 256 - (chunk_size - 1)
                stream.append(header)
                stream.append(val)
                L -= chunk_size
        else:
            # Weniger als 3 gleiche? Ab in den Einzelgänger-Puffer!
            # Wir hängen den Wert L-mal an (also 1x oder 2x)
            for _ in range(L):
                literal_buffer.append(val)
                
    # Am Ende der Matrix noch den Rest aus dem Puffer wegschreiben
    flush_literals()
    
    # 3. Shannon-Entropie auf diesem hochoptimierten Stream berechnen
    stream_array = np.array(stream)
    _, counts = np.unique(stream_array, return_counts=True)
    probabilities = counts / len(stream_array)
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Bits in Bytes umwandeln
    total_bytes = (entropy * len(stream_array)) / 8.0
    
    return int(np.ceil(total_bytes))

# ==========================================
# 2. HAUPTKOMPRESSIONS-FUNKTION
# ==========================================

def svd_compression(img, n_components, epsilon=0.01, bits=8):    
    """
    Komprimiert ein Bild mittels rSVD, simuliert Quantisierung und RLE.
    Gibt das rekonstruierte Bild UND die berechnete Dateigröße in Bytes zurück.
    """
    img_float = img.astype(np.float64)
    total_bytes = 0
    
    if len(img.shape) == 2:
        # Für Graustufenbilder: Einfach rSVD auf die 2D-Matrix anwenden
        u, s, v = rvd.rand_svd(img_float, n_components)
        
        u_quant, u_scale, _ = quantize_and_threshold(u, epsilon, bits)
        v_quant, v_scale, _ = quantize_and_threshold(v, epsilon, bits)
        
        # Größe berechnen: RLE von U und V + Skalierungsfaktoren + Singulärwerte (float32)
        total_bytes += calculate_rle_huffman_bytes(u_quant) + 4
        total_bytes += calculate_rle_huffman_bytes(v_quant) + 4
        total_bytes += n_components * 4
        
        reconstructed = dequantize(u_quant, u_scale) @ np.diag(s) @ dequantize(v_quant, v_scale)
        
        # Für Farbbilder: rSVD auf jeden Kanal separat anwenden
    elif len(img.shape) == 3:
        reconstructed = np.zeros_like(img_float)
        for channel in range(img.shape[2]):
            u, s, v = rvd.rand_svd(img_float[:, :, channel], n_components)
            
            u_quant, u_scale, _ = quantize_and_threshold(u, epsilon, bits)
            v_quant, v_scale, _ = quantize_and_threshold(v, epsilon, bits)
            
            total_bytes += calculate_rle_huffman_bytes(u_quant) + 4
            total_bytes += calculate_rle_huffman_bytes(v_quant) + 4
            total_bytes += n_components * 4
            
            reconstructed[:, :, channel] = dequantize(u_quant, u_scale) @ np.diag(s) @ dequantize(v_quant, v_scale)
            
    else:
        raise ValueError("Das Eingabebild muss entweder 2D oder 3D sein.")
        
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed, total_bytes

# ==========================================
# 3. PLOTTING UND EVALUATION
# ==========================================

def plot_image_compressions(original_img, k_values, epsilon=0.01):
    """Plottet das Original und verschiedene Kompressionsstufen mit echten KB-Größen."""
    n_plots = 1 + len(k_values)
    cols = 3
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    cmap = 'gray' if len(original_img.shape) == 2 else None
    
    # Originalgröße berechnen (Breite * Höhe * Kanäle in Bytes)
    m, n = original_img.shape[:2]
    c = 1 if len(original_img.shape) == 2 else original_img.shape[2]
    original_kb = (m * n * c) / 1024
    
    axes[0].imshow(original_img, cmap=cmap)
    axes[0].set_title(f"Originalbild\nUnkomprimiert: {original_kb:.1f} KB")
    axes[0].axis('off')
    
    print(f"Berechne {len(k_values)} Bilder für den visuellen Vergleich...")
    for i, k in enumerate(k_values):
        comp_img, total_bytes = svd_compression(original_img, k, epsilon=epsilon)
        comp_kb = total_bytes / 1024
        
        axes[i+1].imshow(comp_img, cmap=cmap)
        axes[i+1].set_title(f"Komprimiert (k = {k})\nGröße: {comp_kb:.1f} KB")
        axes[i+1].axis('off')
        
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_size_vs_mse(original_img, epsilon=0.01):
    """Erstellt den Graphen: Echte Dateigröße (KB) vs. MSE."""
    m, n = original_img.shape[:2]
    c = 1 if len(original_img.shape) == 2 else original_img.shape[2]
    original_bytes = m * n * c
    
    # Wir testen k-Werte in sinnvollen Schritten, bis max. k=300
    k_values = list(range(10, 310, 20)) 
    mses = []
    sizes_kb = []
    
    print("Berechne Daten für den Trade-off Graphen (RLE-Größe vs. MSE)...")
    for k in k_values:
        comp_img, total_bytes = svd_compression(original_img, k, epsilon=epsilon)
        
        mse = np.mean((original_img.astype(np.float64) - comp_img.astype(np.float64)) ** 2)
        
        mses.append(mse)
        sizes_kb.append(total_bytes / 1024)
        
    plt.figure(figsize=(10, 6))
    plt.plot(mses, sizes_kb, marker='o', linestyle='-', color='b', linewidth=2)
    
    # Eine gestrichelte Linie für die Originalgröße als Referenz einzeichnen
    plt.axhline(y=original_bytes/1024, color='r', linestyle='--', label=f'Originalgröße ({original_bytes/1024:.1f} KB)')
    
    plt.xlabel("Fehler: Mean Squared Error (MSE)", fontsize=12)
    plt.ylabel("Dateigröße nach RLE-Kompression (KB)", fontsize=12)
    plt.title(f"Echte Kompressionsleistung (Quantisierung $\epsilon={epsilon}$)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Ein paar k-Werte als Label an die Punkte schreiben
    for i, k in enumerate(k_values):
        if i % 2 == 0 or k == k_values[-1]: 
            plt.annotate(f"k={k}", (mses[i], sizes_kb[i]), 
                         textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)
            
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. MAIN BLOCK
# ==========================================
if __name__ == "__main__":
    # Bild laden
    img_path = r"img\Kirche.tiff"
    pil_img = Image.open(img_path)
    
    # Für schnelle Berechnungen verkleinern
    pil_img.thumbnail((800, 800)) 
    img = np.array(pil_img)
    
    # Alpha-Kanal entfernen, falls vorhanden
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
        
    # Schwellenwert (alles unter 0.05 des Maximalwerts wird auf 0 gesetzt)
    test_epsilon = 0.05
    
    # --- Plot 1: Visueller Vergleich ---
    test_k_values = [10, 50, 150, 250, 350]
    plot_image_compressions(img, test_k_values, epsilon=test_epsilon)
    
    # --- Plot 2: Der Trade-off Graph (KB vs. MSE) ---
    plot_size_vs_mse(img, epsilon=test_epsilon)