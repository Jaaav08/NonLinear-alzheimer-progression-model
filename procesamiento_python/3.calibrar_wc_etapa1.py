"""
Este script calibra automáticamente los parámetros del modelo Wilson–Cowan
(entrada excitatoria P_E, ganancia de acoplamiento G y nivel de ruido)
para un paciente específico y una etapa clínica determinada (CN, EMCI, MCI, AD).

El procedimiento no busca ajustar directamente una señal empírica, sino
identificar un conjunto de parámetros fisiológicamente plausibles que
genere de manera espontánea un perfil espectral por bandas consistente
con patrones reportados en la literatura EEG/MEG para cada etapa.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from scipy.signal import welch


# ===========================================================
# FUNCIONES AUXILIARES
# ===========================================================
def sigmoid(x, a, theta):
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


def normalize_connectome(C):
    C = C.astype(float)
    max_val = np.max(np.abs(C))
    return C / max_val if max_val > 0 else C


# ===========================================================
# MODELO WILSON–COWAN (SIMULACIÓN CORTA)
# ===========================================================
def simulate_wc_once(
    C,
    P_E,
    G,
    noise_std,
    dt_ms=1.0,
    T_ms=5000.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    C = normalize_connectome(C)
    N = C.shape[0]

    tau_E, tau_I = 0.010, 0.020
    w_EE, w_IE, w_EI, w_II = 12.0, 4.0, 13.0, 11.0
    a_E, theta_E = 1.5, 2.5
    a_I, theta_I = 1.5, 3.0

    dt = dt_ms / 1000.0
    n_steps = int((T_ms / 1000.0) / dt)

    E = 0.1 + 0.01 * rng.standard_normal(N)
    I = 0.1 + 0.01 * rng.standard_normal(N)
    E_ts = np.zeros((n_steps, N))

    for k in range(n_steps):
        E_ts[k] = E
        coupling = C @ E

        input_E = w_EE * E - w_EI * I + G * coupling + P_E
        input_I = w_IE * E - w_II * I + 1.0

        input_E += noise_std * np.sqrt(dt) * rng.standard_normal(N)
        input_I += noise_std * np.sqrt(dt) * rng.standard_normal(N)

        S_E = sigmoid(input_E, a_E, theta_E)
        S_I = sigmoid(input_I, a_I, theta_I)

        E += dt * (-E + S_E) / tau_E
        I += dt * (-I + S_I) / tau_I

    return E_ts


# ===========================================================
# ANÁLISIS ESPECTRAL
# ===========================================================
def compute_bandpower(E_ts, fs):
    MEEG = E_ts.T
    MEEG = (MEEG - MEEG.mean(axis=1, keepdims=True)) / MEEG.std(axis=1, keepdims=True)

    freqs, psd = welch(MEEG, fs=fs, axis=1, nperseg=min(int(fs * 2), MEEG.shape[1]))
    bands = [(1,4), (4,8), (8,12), (13,30), (30,80)]

    bp = []
    for fmin, fmax in bands:
        idx = (freqs >= fmin) & (freqs <= fmax)
        bp.append(np.trapz(psd[:, idx], freqs[idx], axis=1).mean())

    bp = np.array(bp)
    return bp / bp.sum() if bp.sum() > 0 else bp


# ===========================================================
# CALIBRACIÓN POR ETAPA
# ===========================================================
def calibrate(stage, connectome, out_file):

    targets = {
        "CN":  np.array([0.10, 0.15, 0.45, 0.20, 0.10]),
        "EMCI":np.array([0.15, 0.22, 0.38, 0.18, 0.07]),
        "MCI": np.array([0.25, 0.35, 0.25, 0.12, 0.03]),
        "AD":  np.array([0.45, 0.35, 0.10, 0.05, 0.05]),
    }

    target = targets[stage] / targets[stage].sum()
    fs = 1000.0

    best_err = np.inf
    best_cfg = None
    best_bp = None

    for P_E in [1.0, 1.2, 1.5, 1.8]:
        for G in [0.6, 0.8, 1.0, 1.2]:
            for noise in [0.05, 0.10, 0.20]:
                E_ts = simulate_wc_once(connectome, P_E, G, noise)
                bp = compute_bandpower(E_ts, fs)
                err = np.sum((bp - target) ** 2)

                if err < best_err:
                    best_err = err
                    best_cfg = (P_E, G, noise)
                    best_bp = bp

    result = {
        "stage": stage,
        "P_E": best_cfg[0],
        "G": best_cfg[1],
        "noise_std": best_cfg[2],
        "bandpower_norm": best_bp.tolist(),
        "target_bandpower_norm": target.tolist(),
        "error": float(best_err),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


# ===========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=str)
    args = parser.parse_args()

    # Línea 152 corregida:
    BASE = Path(r" ")
    C = np.loadtxt(BASE / "subj_connectome_dipy.csv", delimiter=",")

    calibrate(args.stage.upper(), C, BASE / "calibrated_params.json")
