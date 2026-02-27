
"""
Generación de señales M/EEG oscilatorias a partir de la dinámica
Wilson–Cowan, interpretando la actividad excitatoria E(t) como una
envolvente lenta que modula oscilaciones en bandas de frecuencia
(delta–gamma).

NOTA IMPORTANTE:
Este script se utiliza exclusivamente para visualización, análisis
cualitativo y discusión. Los resultados cuantitativos del estudio se
basan en la señal generada mediante el modelo lineal directo.
"""

from pathlib import Path
import json

import numpy as np
from scipy.signal import butter, filtfilt, welch


# ===========================================================
# RUTAS
# ===========================================================
BASE = Path(r" ")
OUT  = BASE / "output"

E_FILE        = OUT / "E_timeseries.npy"
BP_NORM_FILE  = OUT / "bandpower_norm.npy"
MEEG_FILE     = OUT / "MEEG_sim.npy"
PSD_FILE      = OUT / "PSD.npy"
FREQS_FILE    = OUT / "freqs_psd.npy"
FC_FILE       = OUT / "FC.npy"
BP_FILE       = OUT / "bandpower.npy"
META_FILE     = OUT / "meeg_metadata.json"
SIM_META_FILE = OUT / "simulation_metadata.json"


# ===========================================================
# FILTRO PASA-BANDA
# ===========================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


# ===========================================================
# PROCESO PRINCIPAL
# ===========================================================
def main():

    if not E_FILE.exists():
        raise FileNotFoundError(f"No se encontró {E_FILE}")

    E = np.load(E_FILE)
    E_ts = E.T if E.shape[0] > E.shape[1] else E
    n_regions, n_steps = E_ts.shape

    # Frecuencia de muestreo
    fs = 1000.0
    if SIM_META_FILE.exists():
        with open(SIM_META_FILE, "r", encoding="utf-8") as f:
            fs = 1000.0 / json.load(f).get("dt_ms", 1.0)

    # Bandas
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }
    band_names = list(bands.keys())

    # Pesos por banda
    if BP_NORM_FILE.exists():
        bp_norm = np.load(BP_NORM_FILE)
        weights = bp_norm.mean(axis=0)
    else:
        weights = np.array([0.3, 0.3, 0.2, 0.15, 0.05])

    weights /= weights.sum()

    # Envolvente normalizada
    E_env = (E_ts - E_ts.min(axis=1, keepdims=True)) / (
        E_ts.max(axis=1, keepdims=True) - E_ts.min(axis=1, keepdims=True) + 1e-12
    )

    rng = np.random.default_rng(1234)
    MEEG = np.zeros((n_regions, n_steps))

    for r in range(n_regions):
        noise = rng.standard_normal(n_steps)
        signal = np.zeros(n_steps)

        for i, (fmin, fmax) in enumerate(bands.values()):
            band = butter_bandpass_filter(noise, fmin, fmax, fs)
            band /= np.std(band) + 1e-12
            band *= np.sqrt(weights[i])
            signal += band * E_env[r]

        signal += 0.05 * rng.standard_normal(n_steps)
        signal = (signal - signal.mean()) / (signal.std() + 1e-12)
        MEEG[r] = signal

    np.save(MEEG_FILE, MEEG)

    # PSD, FC y bandpower
    freqs, psd = welch(MEEG, fs=fs, nperseg=min(int(2 * fs), n_steps), axis=1)
    np.save(PSD_FILE, psd)
    np.save(FREQS_FILE, freqs)
    np.save(FC_FILE, np.corrcoef(MEEG))

    bp = np.zeros((n_regions, len(bands)))
    for i, (fmin, fmax) in enumerate(bands.values()):
        idx = (freqs >= fmin) & (freqs <= fmax)
        bp[:, i] = np.trapz(psd[:, idx], freqs[idx], axis=1)

    np.save(BP_FILE, bp)
    np.save(BP_NORM_FILE, bp / (bp.sum(axis=1, keepdims=True) + 1e-12))

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Señales M/EEG oscilatorias generadas con fines de visualización.",
                "bands": band_names,
                "fs": fs,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("[OK] Señales M/EEG oscilatorias generadas correctamente.")


if __name__ == "__main__":
    main()
