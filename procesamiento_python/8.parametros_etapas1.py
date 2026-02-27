"""
===========================================================
Este script carga los parámetros calibrados del modelo
neuronal (P_E, G y desviación estándar del ruido) para
diferentes etapas clínicas (CN, EMCI, MCI y AD), y
visualiza su evolución a lo largo de la progresión
de la enfermedad.

Los parámetros son obtenidos a partir del proceso de
calibración del modelo sobre señales M/EEG simuladas.

Este script produce:
Gráficas comparativas de la evolución de parámetros
por etapa clínica.
===========================================================
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===========================================================
# RUTAS BASE POR ETAPA CLÍNICA
# ===========================================================
BASE_CN   = Path(r" ")
BASE_EMCI = Path(r" ")
BASE_MCI  = Path(r" ")
BASE_AD   = Path(r" ")

PATIENTS = {
    "CN": BASE_CN,
    "EMCI": BASE_EMCI,
    "MCI": BASE_MCI,
    "AD": BASE_AD
}

STAGES = ["CN", "EMCI", "MCI", "AD"]

# ===========================================================
# FUNCIÓN PARA CARGAR PARÁMETROS CALIBRADOS
# ===========================================================
def load_calibrated_params(base_path: Path) -> dict:
    """
    Carga los parámetros calibrados del modelo desde un archivo JSON.

    Parámetros
    ----------
    base_path : Path
        Ruta base del sujeto o etapa clínica.

    Retorna
    -------
    dict
        Diccionario con los parámetros calibrados.
    """
    calib_file = base_path / "output" / "calibrated_params.json"
    with open(calib_file, "r", encoding="utf-8") as f:
        return json.load(f)

# ===========================================================
# CARGA DE PARÁMETROS POR ETAPA
# ===========================================================
P_E = []
G = []
NOISE = []

print("\nCargando parámetros calibrados por etapa clínica:\n")

for stage in STAGES:
    params = load_calibrated_params(PATIENTS[stage])

    P_E.append(params["P_E"])
    G.append(params["G"])
    NOISE.append(params["noise_std"])

    print(f"  {stage}: P_E={params['P_E']:.4f}, "
          f"G={params['G']:.4f}, "
          f"noise_std={params['noise_std']:.4f}")

# ===========================================================
# VISUALIZACIÓN
# ===========================================================
plt.figure(figsize=(12, 5))

# --- Subplot 1: Parámetro excitatorio P_E ---
plt.subplot(1, 3, 1)
plt.plot(STAGES, P_E, marker="o", linewidth=2)
plt.title("Parámetro excitatorio $P_E$")
plt.ylabel("Valor")
plt.grid(True)

# --- Subplot 2: Acoplamiento global G ---
plt.subplot(1, 3, 2)
plt.plot(STAGES, G, marker="o", linewidth=2, color="orange")
plt.title("Acoplamiento global $G$")
plt.ylabel("Valor")
plt.grid(True)

# --- Subplot 3: Ruido ---
plt.subplot(1, 3, 3)
plt.plot(STAGES, NOISE, marker="o", linewidth=2, color="red")
plt.title("Desviación estándar del ruido")
plt.ylabel("Valor")
plt.grid(True)

plt.suptitle("Evolución de parámetros calibrados del modelo por etapa clínica")
plt.tight_layout()
plt.show()
