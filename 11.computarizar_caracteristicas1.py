"""
===========================================================
 Descripción:
     Definición de características resumen extraídas de:
     - Bandpower espectral
     - Conectividad funcional (FC)
     - Conectividad estructural (SC)

     Estas métricas se utilizan con fines descriptivos
     y exploratorios, no inferenciales.
===========================================================
"""

import numpy as np
import networkx as nx

# ===========================================================
# FEATURES ESPECTRALES
# ===========================================================

def compute_theta_alpha_ratio(bp_mean: np.ndarray) -> float:
    """Razón theta / alpha, biomarcador clásico de enlentecimiento."""
    theta = bp_mean[1]
    alpha = bp_mean[2]
    return np.nan if alpha == 0 else theta / alpha


def compute_slowing(bp_mean: np.ndarray) -> float:
    """Potencia lenta total (delta + theta)."""
    return bp_mean[0] + bp_mean[1]

# ===========================================================
# FEATURES DE CONECTIVIDAD FUNCIONAL
# ===========================================================

def compute_FC_strength(FC: np.ndarray) -> float:
    """Fuerza media absoluta de conectividad funcional."""
    return np.mean(np.abs(FC))


def compute_FC_clustering(FC: np.ndarray) -> float:
    """Coeficiente de clustering promedio (grafo ponderado)."""
    G = nx.from_numpy_array(FC)
    clustering = nx.clustering(G, weight="weight")
    return np.mean(list(clustering.values()))


def compute_FC_efficiency(FC: np.ndarray) -> float:
    """Eficiencia global de la red funcional."""
    G = nx.from_numpy_array(FC)
    return nx.global_efficiency(G)

# ===========================================================
# FEATURES DE CONECTIVIDAD ESTRUCTURAL
# ===========================================================

def compute_SC_strength(SC: np.ndarray) -> float:
    """Fuerza media del conectoma estructural."""
    return np.mean(SC)


def compute_SC_density(SC: np.ndarray) -> float:
    """Densidad del grafo estructural."""
    num_edges = np.count_nonzero(SC)
    total_possible = SC.shape[0] * (SC.shape[0] - 1)
    return num_edges / total_possible
