"""
Simula la dinámica de una red de poblaciones corticales acopladas
mediante el modelo no lineal de Wilson–Cowan, utilizando como
estructura de acoplamiento un conectoma estructural individual.

Las series de tiempo generadas constituyen la base para la simulación
de señales M/EEG y el análisis no lineal presentado en los resultados.
"""

import json
from pathlib import Path
import numpy as np


# ===========================================================
# FUNCIONES AUXILIARES
# ===========================================================
def sigmoid(x, a, theta):
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


def load_connectome(path: Path):
    C = np.loadtxt(path, delimiter=",")
    if C.shape[0] != C.shape[1]:
        raise ValueError("El conectoma debe ser una matriz cuadrada.")
    return C.astype(float)


def normalize_connectome(C):
    max_val = np.max(np.abs(C))
    return C / max_val if max_val > 0 else C


# ===========================================================
# PARÁMETROS DEL MODELO
# ===========================================================
def default_params(N, calib=None):
    params = dict(
        tau_E=0.010,
        tau_I=0.020,
        w_EE=12.0,
        w_IE=4.0,
        w_EI=13.0,
        w_II=11.0,
        a_E=1.5,
        theta_E=2.5,
        a_I=1.5,
        theta_I=3.0,
        G=1.2,
        sigma_E=0.3,
        sigma_I=0.3,
        P_E=1.5 * np.ones(N),
        P_I=1.0 * np.ones(N),
    )

    if calib is not None:
        params["P_E"] = calib["P_E"] * np.ones(N)
        params["G"] = calib["G"]
        params["sigma_E"] = calib["noise_std"]
        params["sigma_I"] = calib["noise_std"]

    return params


# ===========================================================
# PASO DE DINÁMICA WILSON–COWAN
# ===========================================================
def wc_step(E, I, C, params, dt, rng):
    coupling = C @ E

    input_E = (
        params["w_EE"] * E
        - params["w_EI"] * I
        + params["G"] * coupling
        + params["P_E"]
    )
    input_I = (
        params["w_IE"] * E
        - params["w_II"] * I
        + params["P_I"]
    )

    input_E += params["sigma_E"] * np.sqrt(dt) * rng.standard_normal(E.shape)
    input_I += params["sigma_I"] * np.sqrt(dt) * rng.standard_normal(I.shape)

    S_E = sigmoid(input_E, params["a_E"], params["theta_E"])
    S_I = sigmoid(input_I, params["a_I"], params["theta_I"])

    dE = (-E + S_E) / params["tau_E"]
    dI = (-I + S_I) / params["tau_I"]

    return E + dt * dE, I + dt * dI


# ===========================================================
# SIMULACIÓN PRINCIPAL
# ===========================================================
def simulate(connectome_path, outdir, dt_ms=1.0, T_ms=8000.0, seed=42):

    outdir.mkdir(parents=True, exist_ok=True)

    C = normalize_connectome(load_connectome(connectome_path))
    N = C.shape[0]

    calib_path = outdir / "calibrated_params.json"
    calib = json.load(open(calib_path)) if calib_path.exists() else None

    params = default_params(N, calib)

    dt = dt_ms / 1000.0
    n_steps = int((T_ms / 1000.0) / dt)

    rng = np.random.default_rng(seed)
    E = 0.1 + 0.01 * rng.standard_normal(N)
    I = 0.1 + 0.01 * rng.standard_normal(N)

    E_ts = np.zeros((n_steps, N))
    I_ts = np.zeros((n_steps, N))

    for k in range(n_steps):
        E_ts[k] = E
        I_ts[k] = I
        E, I = wc_step(E, I, C, params, dt, rng)

    np.save(outdir / "E_timeseries.npy", E_ts)
    np.save(outdir / "I_timeseries.npy", I_ts)

    return E_ts, I_ts

# ===========================================================
# BLOQUE DE EJECUCIÓN
# ===========================================================
if __name__ == "__main__":
    # 1. Definimos la ruta base (usando r"..." para evitar errores de Windows)
    BASE = Path(r" ")
    
    # 2. Definimos el archivo del conectoma (el que generaste con DIPY)
    CONECTOMA_FILE = BASE / "subj_connectome_dipy.csv"
    
    # 3. Llamamos a la función de simulación
    print("Iniciando simulación de Wilson-Cowan...")
    E_ts, I_ts = simulate(
        connectome_path=CONECTOMA_FILE,
        outdir=BASE,
        T_ms=8000.0  # Simulación de 8 segundos
    )
    
    print(f"Simulación completada. Archivos guardados en: {BASE}")