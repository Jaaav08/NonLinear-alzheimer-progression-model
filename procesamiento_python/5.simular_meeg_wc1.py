#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_meeg_from_wc.py

Generación de señales tipo M/EEG a partir de la actividad excitatoria
E(t) producida por el modelo Wilson–Cowan acoplado a un conectoma estructural.

Este script implementa un operador de observación lineal simplificado
(leadfield identidad) y se utiliza para el cálculo de biomarcadores
funcionales cuantitativos:

- Densidad espectral de potencia (PSD)
- Potencia por bandas (bandpower)
- Conectividad funcional (correlación de Pearson)

NOTA METODOLÓGICA:
Este procedimiento preserva una relación directa entre la dinámica
no lineal del modelo y la señal observada, y constituye la base de los
resultados cuantitativos reportados en la tesis.
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch, butter, filtfilt


# ===========================================================
# 1. RUTAS
# ===========================================================
BASE = Path(r" ")
OUT  = BASE / "output"

E_FILE        = OUT / "E_timeseries.npy"
MEEG_FILE     = OUT / "MEEG_sim.npy"
PSD_FILE      = OUT / "PSD.npy"
FREQS_FILE    = OUT / "freqs_psd.npy"
FC_FILE       = OUT / "FC.npy"
BP_FILE       = OUT / "bandpower.npy"
BP_NORM_FILE  = OUT / "bandpower_norm.npy"
META_FILE     = OUT / "meeg_metadata.json"
SIM_META_FILE = OUT / "simulation_metadata.json"


# ===========================================================
# 2. UTILIDADES
# ===========================================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Filtro pasa-bajas Butterworth aplicado por canal.
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype="low")
    return filtfilt(b, a, data, axis=1)


def compute_bandpower(psd, freqs):
    """
    Calcula la potencia espectral integrada por banda de frecuencia.
    """
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }

    n_channels, _ = psd.shape
    bp = np.zeros((n_channels, len(bands)))
    band_names = list(bands.keys())

    for i, (name, (fmin, fmax)) in enumerate(bands.items()):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if np.any(idx):
            bp[:, i] = np.trapz(psd[:, idx], freqs[idx], axis=1)

    return bp, band_names


# ===========================================================
# 3. PROCESO PRINCIPAL
# ===========================================================
def main():

    # -------------------------------------------------------
    # 3.1 Cargar actividad excitatoria E(t)
    # -------------------------------------------------------
    if not E_FILE.exists():
        raise FileNotFoundError(f"No se encontró {E_FILE}")

    E = np.load(E_FILE)  # esperado [n_steps, n_regiones]

    # Forzar formato [n_regiones, n_steps]
    E_ts = E.T if E.shape[0] > E.shape[1] else E
    n_regions, n_steps = E_ts.shape

    print(f"E_timeseries cargado: {E.shape} → usado como {E_ts.shape}")

    # -------------------------------------------------------
    # 3.2 Frecuencia de muestreo
    # -------------------------------------------------------
    fs = 1000.0
    if SIM_META_FILE.exists():
        with open(SIM_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fs = 1000.0 / meta.get("dt_ms", 1.0)

    print(f"Frecuencia de muestreo: fs = {fs:.2f} Hz")

    # -------------------------------------------------------
    # 3.3 Operador de observación (leadfield identidad)
    # -------------------------------------------------------
    L = np.eye(n_regions)
    MEEG = L @ E_ts

    # Normalización por canal
    MEEG = (MEEG - MEEG.mean(axis=1, keepdims=True)) / MEEG.std(axis=1, keepdims=True)

    # -------------------------------------------------------
    # 3.4 Filtrado pasa-bajas
    # -------------------------------------------------------
    cutoff = 45.0
    MEEG_filt = butter_lowpass_filter(MEEG, cutoff, fs)

    np.save(MEEG_FILE, MEEG_filt)
    print(f"[OK] MEEG guardado en {MEEG_FILE}")

    # -------------------------------------------------------
    # 3.5 PSD
    # -------------------------------------------------------
    nperseg = min(int(2.0 * fs), n_steps)
    freqs, psd = welch(MEEG_filt, fs=fs, nperseg=nperseg, axis=1)

    np.save(PSD_FILE, psd)
    np.save(FREQS_FILE, freqs)

    # -------------------------------------------------------
    # 3.6 Conectividad funcional
    # -------------------------------------------------------
    FC = np.corrcoef(MEEG_filt)
    np.save(FC_FILE, FC)

    # -------------------------------------------------------
    # 3.7 Bandpower
    # -------------------------------------------------------
    bp, band_names = compute_bandpower(psd, freqs)
    np.save(BP_FILE, bp)

    bp_norm = bp / (bp.sum(axis=1, keepdims=True) + 1e-12)
    np.save(BP_NORM_FILE, bp_norm)

    print("Bandpower normalizado promedio:")
    print(f"{band_names} → {bp_norm.mean(axis=0)}")

    # -------------------------------------------------------
    # 3.8 Metadata
    # -------------------------------------------------------
    meeg_meta = {
        "fs": fs,
        "cutoff_lowpass_Hz": cutoff,
        "bands": band_names,
        "description": (
            "Señal M/EEG generada mediante un operador lineal directo "
            "aplicado a la actividad excitatoria del modelo Wilson–Cowan."
        ),
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meeg_meta, f, indent=4, ensure_ascii=False)

    print("[OK] Proceso finalizado correctamente.")


if __name__ == "__main__":
    main()
