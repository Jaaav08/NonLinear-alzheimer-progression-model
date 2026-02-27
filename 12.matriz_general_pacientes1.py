"""
matriz_general_pacientes.py

Genera una matriz general de características a nivel de sujeto
para 16 pacientes (CN, EMCI, MCI, AD), utilizada como entrada
para el entrenamiento de la red neuronal (script 16).

Incluye:
- Parámetros calibrados del modelo neuronal
- Bandpower normalizado por banda
- Métricas derivadas (theta/alpha, slowing)
- Métricas de conectividad funcional (FC)
- Métricas de conectividad estructural (SC)
- Etiqueta clínica y severidad ordinal

Salida:
- caracteristicas_pacientes_final.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from computarizar_caracteristicas1 import (
    compute_theta_alpha_ratio,
    compute_slowing,
    compute_FC_strength,
    compute_FC_clustering,
    compute_FC_efficiency,
    compute_SC_strength,
    compute_SC_density
)

# =========================================================
# 1. DEFINICIÓN DE LOS 16 PACIENTES
# =========================================================

PATIENTS = {
    # ---------- CN ----------
    "CN_1": {"stage": "CN", "base": Path(r" ")},
    "CN_2": {"stage": "CN", "base": Path(r" ")},
    "CN_3": {"stage": "CN", "base": Path(r" ")},
    "CN_4": {"stage": "CN", "base": Path(r" ")},

    # ---------- EMCI ----------
    "EMCI_1": {"stage": "EMCI", "base": Path(r" ")},
    "EMCI_2": {"stage": "EMCI", "base": Path(r" ")},
    "EMCI_3": {"stage": "EMCI", "base": Path(r" ")},
    "EMCI_4": {"stage": "EMCI", "base": Path(r" ")},

    # ---------- MCI ----------
    "MCI_1": {"stage": "MCI", "base": Path(r" ")},
    "MCI_2": {"stage": "MCI", "base": Path(r" ")},
    "MCI_3": {"stage": "MCI", "base": Path(r" ")},
    "MCI_4": {"stage": "MCI", "base": Path(r" ")},

    # ---------- AD ----------
    "AD_1": {"stage": "AD", "base": Path(r" ")},
    "AD_2": {"stage": "AD", "base": Path(r" ")},
    "AD_3": {"stage": "AD", "base": Path(r" ")},
    "AD_4": {"stage": "AD", "base": Path(r" ")},
}

STAGE_TO_INDEX = {"CN": 0, "EMCI": 1, "MCI": 2, "AD": 3}

# =========================================================
# 2. RUTA DE SALIDA (EXPLÍCITA Y SEGURA)
# =========================================================

OUT_CSV = Path(
    r" "
)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# 3. CONSTRUCCIÓN DE LA MATRIZ DE FEATURES
# =========================================================

rows = []

for subject_id, meta in PATIENTS.items():

    stage = meta["stage"]
    severity_index = STAGE_TO_INDEX[stage]
    out = meta["base"] / "output"

    # ---------- cargar datos ----------
    calib = json.load(open(out / "calibrated_params.json", "r"))
    bp = np.load(out / "bandpower_norm.npy")
    bp_mean = bp.mean(axis=0)

    FC = np.load(out / "FC.npy")
    SC = np.loadtxt(out / "subj_connectome_dipy.csv", delimiter=",")

    BP_delta, BP_theta, BP_alpha, BP_beta, BP_gamma = bp_mean

    row = {
        "subject_id": subject_id,

        # Parámetros calibrados
        "P_E": calib.get("P_E"),
        "G": calib.get("G"),
        "noise_std": calib.get("noise_std"),
        "best_error": calib.get("best_error", calib.get("error")),

        # Bandpower
        "BP_delta": BP_delta,
        "BP_theta": BP_theta,
        "BP_alpha": BP_alpha,
        "BP_beta": BP_beta,
        "BP_gamma": BP_gamma,

        # EEG features
        "theta_alpha_ratio": compute_theta_alpha_ratio(bp_mean),
        "slowing": compute_slowing(bp_mean),

        # FC features
        "FC_strength": compute_FC_strength(FC),
        "FC_clustering": compute_FC_clustering(FC),
        "FC_efficiency": compute_FC_efficiency(FC),

        # SC features
        "SC_strength": compute_SC_strength(SC),
        "SC_density": compute_SC_density(SC),

        # Labels
        "stage": stage,
        "severity_index": severity_index,
    }

    rows.append(row)

# =========================================================
# 4. GUARDAR CSV FINAL
# =========================================================

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("\n==============================")
print("MATRIZ GENERAL GENERADA")
print("==============================")
print(f"Archivo creado en:\n{OUT_CSV.resolve()}")
print(f"Número de sujetos: {df.shape[0]}")
print(f"Número de features: {df.shape[1]}")
print("\nListo para entrenamiento con la red neuronal (script 16).")
