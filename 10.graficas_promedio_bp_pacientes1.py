#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===========================================================
Descripción:
     Análisis comparativo del bandpower normalizado obtenido
     a partir de señales M/EEG simuladas, considerando:

     (1) Promedios por etapa clínica (CN, EMCI, MCI, AD)
     (2) Visualización individual por paciente
     (3) Estadísticos descriptivos de parámetros calibrados
     (4) Medidas globales de SC y FC (no inferenciales)

     Este script tiene carácter descriptivo-comparativo y
     complementa los resultados principales del modelo.

 Salidas:
     - Figuras comparativas de bandpower
     - Resúmenes numéricos impresos en consola
===========================================================
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# CONFIGURACIÓN GENERAL
# ===========================================================

STAGES = ["CN", "EMCI", "MCI", "AD"]
BAND_LABELS = ["delta", "theta", "alpha", "beta", "gamma"]

STAGE_COLORS = {
    "CN": "#1f77b4",    # Azul
    "EMCI": "#ff7f0e",  # Naranja
    "MCI": "#2ca02c",   # Verde
    "AD": "#d62728"     # Rojo
}

# ===========================================================
# LISTA DE PACIENTES
# ===========================================================

ALL_PATIENTS = {
    "CN_1": Path(r" "),
    "CN_2": Path(r" "),
    "CN_3": Path(r" "),
    "CN_4": Path(r" "),

    "EMCI_1": Path(r" "),
    "EMCI_2": Path(r" "),
    "EMCI_3": Path(r" "),
    "EMCI_4": Path(r" "),

    "MCI_1": Path(r" "),
    "MCI_2": Path(r" "),
    "MCI_3": Path(r" "),
    "MCI_4": Path(r" "),

    "AD_1": Path(r" "),
    "AD_2": Path(r" "),
    "AD_3": Path(r" "),
    "AD_4": Path(r" "),
}

# ===========================================================
# FUNCIONES AUXILIARES
# ===========================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_upper_triangle(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices_from(mat, k=1)]


def load_patient_data(name: str, basepath: Path) -> dict:
    output = basepath / "output"
    return {
        "name": name,
        "stage": name.split("_")[0],
        "calib": load_json(output / "calibrated_params.json"),
        "bandpower": np.load(output / "bandpower_norm.npy"),
        "fc": np.load(output / "FC.npy"),
        "sc": np.loadtxt(output / "subj_connectome_dipy.csv", delimiter=","),
    }


def group_by_stage(data: list) -> dict:
    grouped = {stage: [] for stage in STAGES}
    for p in data:
        grouped[p["stage"]].append(p)
    return grouped

# ===========================================================
# ANÁLISIS DESCRIPTIVO
# ===========================================================

def summarize_calibrated_parameters(grouped: dict):
    print("\n======== PARÁMETROS CALIBRADOS (DESCRIPTIVO) ========\n")
    for stage in STAGES:
        P = [p["calib"]["P_E"] for p in grouped[stage]]
        G = [p["calib"]["G"] for p in grouped[stage]]
        N = [p["calib"]["noise_std"] for p in grouped[stage]]

        print(f"{stage}:")
        print(f"  P_E   mean = {np.mean(P):.3f}")
        print(f"  G     mean = {np.mean(G):.3f}")
        print(f"  noise mean = {np.mean(N):.3f}\n")

# ===========================================================
# FIGURA 1: BANDPOWER PROMEDIO POR ETAPA
# ===========================================================

def plot_bandpower_by_stage(grouped: dict):
    plt.figure(figsize=(11, 5))
    width = 0.18
    x = np.arange(len(BAND_LABELS))

    for i, stage in enumerate(STAGES):
        mean_bp = np.mean(
            [p["bandpower"].mean(axis=0) for p in grouped[stage]],
            axis=0
        )
        plt.bar(x + (i - 1.5) * width, mean_bp,
                width, label=stage, color=STAGE_COLORS[stage])

    plt.xticks(x, BAND_LABELS)
    plt.ylabel("Bandpower normalizado promedio")
    plt.title("Bandpower promedio por etapa clínica")
    plt.legend()
    plt.tight_layout()

# ===========================================================
# FIGURA 2: BANDPOWER POR PACIENTE
# ===========================================================

def plot_bandpower_by_patient(data: list):
    fig, ax = plt.subplots(figsize=(16, 7))
    width = 0.05
    x = np.arange(len(BAND_LABELS))

    for i, p in enumerate(data):
        offset = (i - (len(data) / 2)) * width
        bp = p["bandpower"].mean(axis=0)
        color = STAGE_COLORS[p["stage"]]

        ax.bar(x + offset, bp, width, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_ylabel("Bandpower normalizado promedio")
    ax.set_title("Distribución de bandpower por paciente")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=STAGE_COLORS[s]) for s in STAGES
    ]
    ax.legend(legend_handles, STAGES, title="Etapa clínica")
    plt.tight_layout()

# ===========================================================
# MAIN
# ===========================================================

def main():
    all_data = [
        load_patient_data(name, path)
        for name, path in ALL_PATIENTS.items()
    ]

    grouped = group_by_stage(all_data)

    summarize_calibrated_parameters(grouped)
    plot_bandpower_by_stage(grouped)
    plot_bandpower_by_patient(all_data)

    plt.show()

    print("\n======== ANÁLISIS COMPLEMENTARIO COMPLETADO ========\n")


if __name__ == "__main__":
    main()
