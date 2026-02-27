"""
Entrenamiento de una red neuronal multicapa (MLP) para la estimación
del índice ordinal de severidad (0–3: CN, EMCI, MCI, AD) a partir de
biomarcadores simulados derivados de modelos dinámicos cerebrales.

Este script:
- Carga un conjunto de características a nivel de sujeto (CSV)
- Escala las variables usando StandardScaler
- Entrena un clasificador MLP
- Evalúa mediante validación cruzada estratificada
- Entrena un modelo final sobre todo el dataset
- Guarda el modelo y el escalador para uso posterior (interfaz gráfica)

NOTA:
Este código se utiliza como backend de la interfaz gráfica del sistema,
por lo que su funcionalidad y estructura de salida no deben alterarse.

"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# =========================================================
# 1. RUTAS Y CONFIGURACIÓN
# =========================================================

DATA_CSV = Path(
    r" "
)

OUT_DIR = Path(
    r" "
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUT_DIR / "1severity_mlp.pkl"
SCALER_FILE = OUT_DIR / "1severity_scaler.pkl"

# Columnas que NO se utilizan como características de entrada
NON_FEATURE_COLS = [
    "subject_id",
    "stage",
    "severity_index",
    "label"
]

# Etiquetas descriptivas para los reportes
TARGET_NAMES = [
    "CN (0)",
    "EMCI (1)",
    "MCI (2)",
    "AD (3)"
]


# =========================================================
# 2. CARGA DEL DATASET
# =========================================================

def load_dataset(csv_path: Path):
    """
    Carga el dataset de características desde un archivo CSV.

    Parameters
    ----------
    csv_path : Path
        Ruta al archivo CSV con las características por sujeto.

    Returns
    -------
    X : np.ndarray
        Matriz de características (N sujetos × M features).
    y : np.ndarray
        Vector de etiquetas ordinales (0–3).
    feature_cols : list
        Lista de nombres de las características utilizadas.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    # Variable objetivo (ordinal)
    y = df["severity_index"].values.astype(int)

    # Selección automática de features numéricas
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values.astype(float)

    print("\n==============================")
    print("DATASET CARGADO")
    print("==============================")
    print(f"Ruta del archivo : {csv_path}")
    print(f"Número de sujetos: {X.shape[0]}")
    print(f"Número de features: {X.shape[1]}")
    print(f"Clases presentes : {np.unique(y)}")
    print("==============================\n")

    return X, y, feature_cols


# =========================================================
# 3. DEFINICIÓN DEL MODELO
# =========================================================

def build_model(input_dim: int):
    """
    Construye un pipeline de preprocesamiento y clasificación.

    El pipeline incluye:
    - Estandarización de las variables
    - Clasificador MLP con regularización aumentada
      (adecuado para datasets pequeños)

    Parameters
    ----------
    input_dim : int
        Dimensión del vector de entrada.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline completo de escalado + clasificación.
    """

    mlp = MLPClassifier(
        hidden_layer_sizes=(8, 4),
        activation="relu",
        solver="adam",
        alpha=1e-2,
        learning_rate="adaptive",
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp)
    ])

    return pipeline


# =========================================================
# 4. VALIDACIÓN CRUZADA
# =========================================================

def cross_validate_model(X, y, pipeline, n_splits=4):
    """
    Realiza validación cruzada estratificada.

    Parameters
    ----------
    X : np.ndarray
        Matriz de características.
    y : np.ndarray
        Vector de etiquetas.
    pipeline : Pipeline
        Pipeline de entrenamiento.
    n_splits : int
        Número de particiones para la validación cruzada.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    accs = []

    print("\n==============================")
    print("VALIDACIÓN CRUZADA")
    print("==============================\n")

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Pipeline(pipeline.steps)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)

        print(f"Fold {i} | Accuracy: {acc:.3f}")
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print()

    print(f"Accuracy medio CV: {np.mean(accs):.3f}")
    print(f"Desviación estándar: {np.std(accs):.3f}")
    print()

    return accs


# =========================================================
# 5. ENTRENAMIENTO FINAL Y GUARDADO
# =========================================================

def train_final_model(X, y, pipeline):
    """
    Entrena el modelo final usando todo el dataset
    y guarda el clasificador y el escalador entrenados.
    """

    print("\n==============================")
    print("ENTRENAMIENTO FINAL")
    print("==============================\n")

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    print("Accuracy sobre el conjunto completo:", accuracy_score(y, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y, y_pred, digits=3, target_names=TARGET_NAMES))
    print("Matriz de confusión global:")
    print(confusion_matrix(y, y_pred))
    print()

    joblib.dump(pipeline.named_steps["mlp"], MODEL_FILE)
    joblib.dump(pipeline.named_steps["scaler"], SCALER_FILE)

    print(f"Modelo guardado en : {MODEL_FILE}")
    print(f"Scaler guardado en : {SCALER_FILE}\n")


# =========================================================
# MAIN
# =========================================================

def main():
    try:
        X, y, _ = load_dataset(DATA_CSV)
        pipeline = build_model(input_dim=X.shape[1])

        cross_validate_model(X, y, pipeline, n_splits=4)
        train_final_model(X, y, pipeline)

    except FileNotFoundError as e:
        print("\nERROR: Archivo no encontrado")
        print(e)

    except ValueError as e:
        print("\nERROR: Problema con las características numéricas")
        print("Revisa que no existan columnas de texto en las features.")
        print(f"Detalle: {e}")


if __name__ == "__main__":
    main()
