
"""
Generación de figuras para el análisis y presentación de resultados
a partir de señales M/EEG simuladas mediante el modelo Wilson–Cowan.

Este script produce:
1. Señales M/EEG multicanal (ventana temporal corta)
2. Densidad espectral de potencia (PSD) por sensor
3. Potencia promedio por bandas de frecuencia (bandpower)
4. Matriz de conectividad funcional (FC)

Las figuras generadas se utilizan para:
- Análisis exploratorio
- Presentación de resultados
- Discusión fisiológica y clínica (MCI / AD)

NOTA:
Este script no realiza procesamiento adicional; únicamente visualiza
resultados previamente calculados y guardados en disco.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ===========================================================
# 1. RUTAS
# ===========================================================
BASE = Path(r" ")
OUT  = BASE / "output"

MEEG_FILE     = OUT / "MEEG_sim.npy"
PSD_FILE      = OUT / "PSD.npy"
FREQS_FILE    = OUT / "freqs_psd.npy"
FC_FILE       = OUT / "FC.npy"
BP_FILE       = OUT / "bandpower.npy"
BP_NORM_FILE  = OUT / "bandpower_norm.npy"

CALIB_FILE    = OUT / "calibrated_params.json"
SIM_META_FILE = OUT / "simulation_metadata.json"


# ===========================================================
# 2. CARGA DE ARCHIVOS
# ===========================================================
MEEG  = np.load(MEEG_FILE)   # [n_sensores, n_pasos]
PSD   = np.load(PSD_FILE)    # [n_sensores, n_freqs]
freqs = np.load(FREQS_FILE) # [n_freqs]
FC    = np.load(FC_FILE)     # [n_sensores, n_sensores]

n_sensors, n_steps = MEEG.shape

print(f"MEEG cargado: {MEEG.shape}")
print(f"PSD cargada: {PSD.shape}")
print(f"FC cargada: {FC.shape}")


# ===========================================================
# 3. METADATA (etapa clínica y frecuencia de muestreo)
# ===========================================================
stage = "desconocida"
if CALIB_FILE.exists():
    try:
        with open(CALIB_FILE, "r", encoding="utf-8") as f:
            calib = json.load(f)
        stage = calib.get("stage", stage)
    except Exception:
        pass

fs = 1000.0
if SIM_META_FILE.exists():
    try:
        with open(SIM_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fs = 1000.0 / meta.get("dt_ms", 1.0)
    except Exception:
        pass

t = np.arange(n_steps) / fs

print(f"Etapa clínica (calibración): {stage}")
print(f"Frecuencia de muestreo usada en plots: fs = {fs:.2f} Hz")


# ===========================================================
# 4. SEÑAL M/EEG MULTICANAL (ventana temporal corta)
# ===========================================================
plt.figure(figsize=(12, 6))

offset = 0.8
t_window = t < 2.0  # primeros 2 segundos

for i in range(n_sensors):
    plt.plot(
        t[t_window],
        MEEG[i, t_window] + i * offset,
        linewidth=1.0
    )

plt.title(f"Señales M/EEG simuladas – primeros 2 s (Etapa: {stage})")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud normalizada + offset")
plt.tight_layout()
plt.show()


# ===========================================================
# 5. PSD POR SENSOR (escala logarítmica)
# ===========================================================
plt.figure(figsize=(12, 6))

for i in range(n_sensors):
    plt.semilogy(freqs, PSD[i])

plt.xlim(0, 40)
plt.title(f"Densidad espectral de potencia (PSD) por sensor (Etapa: {stage})")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia (escala logarítmica)")
plt.tight_layout()
plt.show()


# ===========================================================
# 6. POTENCIA PROMEDIO POR BANDA
# ===========================================================
use_normalized = BP_NORM_FILE.exists()

if use_normalized:
    bandpower = np.load(BP_NORM_FILE)
    ylabel = "Fracción de potencia"
    print("Usando bandpower normalizado para visualización.")
else:
    bandpower = np.load(BP_FILE)
    ylabel = "Potencia integrada (unidades relativas)"
    print("Usando bandpower crudo para visualización.")

bands = [
    "Delta (1–4 Hz)",
    "Theta (4–8 Hz)",
    "Alpha (8–12 Hz)",
    "Beta (13–30 Hz)",
    "Gamma (30–80 Hz)",
]

bp_mean = bandpower.mean(axis=0)

plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(bands)), bp_mean)
plt.xticks(np.arange(len(bands)), bands, rotation=15)
plt.ylabel(ylabel)
plt.title(f"Bandpower promedio por banda de frecuencia (Etapa: {stage})")
plt.tight_layout()
plt.show()

print("Bandpower promedio mostrado en el gráfico:")
print(f"{bands} → {bp_mean}")


# ===========================================================
# 7. MATRIZ DE CONECTIVIDAD FUNCIONAL
# ===========================================================
plt.figure(figsize=(7, 6))
plt.imshow(FC, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Correlación de Pearson")
plt.title(f"Conectividad funcional (FC) – Etapa: {stage}")
plt.xlabel("Sensor / Región")
plt.ylabel("Sensor / Región")
plt.tight_layout()
plt.show()
