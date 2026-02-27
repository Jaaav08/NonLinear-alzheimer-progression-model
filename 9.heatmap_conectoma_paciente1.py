"""
===========================================================
 Descripción:
     Este script visualiza la matriz de conectividad estructural
     (SC) de un paciente individual, obtenida a partir de datos
     de difusión (DIPY).

     La matriz SC constituye la base estructural del modelo
     neuronal empleado en la simulación de señales M/EEG.

 Salida:
     - Mapa de calor (heatmap) de la conectividad estructural.
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===========================================================
# RUTA AL CONECTOMA ESTRUCTURAL DEL PACIENTE
# ===========================================================
BASE = Path(r" ")
SC_FILE = BASE / "output" / "subj_connectome_dipy.csv"

# ===========================================================
# CARGA DEL CONECTOMA
# ===========================================================
SC = np.loadtxt(SC_FILE, delimiter=",")

print(f"Conectoma estructural cargado: {SC.shape[0]} regiones")

# ===========================================================
# VISUALIZACIÓN
# ===========================================================
plt.figure(figsize=(6, 5))
plt.imshow(SC, cmap="hot", interpolation="nearest")
plt.colorbar(label="Fuerza de conexión estructural")
plt.title("Matriz de conectividad estructural (SC)")
plt.xlabel("Regiones")
plt.ylabel("Regiones")
plt.tight_layout()
plt.show()
