
"""
Interfaz gráfica con Streamlit para explorar el modelo de
predicción de severidad del Alzheimer a partir de biomarcadores
simulados (conectoma + Wilson–Cowan + M/EEG sintética).

Flujo tipo wizard:

1. Datos del paciente
2. Biomarcadores del modelo (16 features numéricos)
3. Resultados del modelo y alerta clínica (tabs):
   - Resumen Ejecutivo
   - Perfil Espectral (Comparativa)
   - Probabilidades por Etapa
   - Detalles Numéricos
4. Descarga de informe en PDF

"""

import numpy as np
import pandas as pd
from pathlib import Path
import io

import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Configuración de estilo global para gráficos (estilo científico, serif)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "Times New Roman", "Times", "serif"]
plt.rcParams["axes.edgecolor"] = "#4a5568"
plt.rcParams["axes.labelcolor"] = "#2d3748"
plt.rcParams["xtick.color"] = "#2d3748"
plt.rcParams["ytick.color"] = "#2d3748"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

# PDF
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# =========================================================
# 1. RUTAS A MODELO Y SCALER
# =========================================================

BASE_DIR = Path(r" ")
MODEL_FILE = BASE_DIR / "1severity_mlp.pkl"
SCALER_FILE = BASE_DIR / "1severity_scaler.pkl"

STAGE_LABELS = {
    0: "CN",
    1: "EMCI",
    2: "MCI",
    3: "AD"
}

# Orden EXACTO de los biomarcadores usado en el entrenamiento
FEATURE_NAMES = [
    "P_E",
    "G",
    "noise_std",
    "best_error",
    "BP_delta",
    "BP_theta",
    "BP_alpha",
    "BP_beta",
    "BP_gamma",
    "theta_alpha_ratio",
    "slowing",
    "FC_strength",
    "FC_clustering",
    "FC_efficiency",
    "SC_strength",
    "SC_density",
]


# =========================================================
# 2. CARGA DE MODELO (CACHEADO)
# =========================================================

@st.cache_resource
def load_model_and_scaler():
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"No se encontraron MODEL_FILE o SCALER_FILE en {MODEL_FILE.parent}"
        )

    mlp = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return mlp, scaler


def compute_risk_zone(proba, severity_soft):
    """
    Asigna una zona tipo semáforo según probabilidades y severidad suave.

    proba: array [p_CN, p_EMCI, p_MCI, p_AD]
    severity_soft: sum_k k * p_k
    """
    p_cn, p_emci, p_mci, p_ad = proba

    # Zona roja: alta prob. de AD o severidad elevada
    if (p_ad >= 0.5) or (severity_soft >= 2.2):
        zone = "Zona roja"
        desc = (
            "Perfil compatible con fase avanzada o alto riesgo. "
            "Sugiere una alteración marcada de la dinámica cortical. "
            "Este resultado NO es diagnóstico, pero podría motivar "
            "una evaluación clínica detallada."
        )
    # Zona amarilla: riesgo moderado
    elif (p_emci >= 0.4) or (p_mci >= 0.4) or (severity_soft >= 1.0):
        zone = "Zona amarilla"
        desc = (
            "Patrón compatible con deterioro leve o moderado. "
            "Podría representar un estado transicional (EMCI / MCI). "
            "Se recomienda correlacionar con historia clínica y pruebas neuropsicológicas."
        )
    # Zona verde: patrón relativamente conservado
    else:
        zone = "Zona verde"
        desc = (
            "Perfil más cercano a controles sanos o cambios muy leves. "
            "Los biomarcadores no muestran un patrón fuertemente alterado, "
            "aunque el seguimiento clínico sigue siendo fundamental."
        )

    return zone, desc


def build_pdf_report(
    patient_info,
    feature_dict,
    stage_pred,
    severity_soft,
    proba,
    risk_zone,
    risk_text,
):
    """
    Construye un PDF simple con resumen del paciente y del modelo.
    Devuelve los bytes del PDF.
    IMPORTANTE: solo usa caracteres ASCII / Latin-1 (sin ≈ ni guiones raros).
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)

    pdf.cell(0, 10, "Informe de biomarcadores y riesgo de Alzheimer", ln=True)
    pdf.ln(4)

    # Datos del paciente
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Datos del paciente", ln=True)
    pdf.set_font("Arial", "", 11)

    pdf.multi_cell(0, 6, "Codigo: {}".format(patient_info.get("id", "")))
    pdf.multi_cell(0, 6, "Nombre: {}".format(patient_info.get("name", "")))
    pdf.multi_cell(0, 6, "Sexo: {}".format(patient_info.get("sex", "")))
    pdf.multi_cell(0, 6, "Edad: {}".format(patient_info.get("age", "")))
    pdf.multi_cell(
        0,
        6,
        "Escolaridad (anios): {}".format(patient_info.get("schooling", "")),
    )
    pdf.multi_cell(0, 6, "MMSE: {}".format(patient_info.get("mmse", "")))
    pdf.multi_cell(0, 6, "CDR: {}".format(patient_info.get("cdr", "")))
    pdf.ln(4)

    obs = patient_info.get("notes", "").strip()
    if obs:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Notas clinicas / observaciones", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, obs)
        pdf.ln(4)

    # Resultados del modelo
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Resultados del modelo", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Etapa clinica predicha: {} (indice ordinal aprox. {:.2f})".format(
            stage_pred, severity_soft
        ),
    )
    pdf.multi_cell(0, 6, "Zona de riesgo: {}".format(risk_zone))
    pdf.multi_cell(0, 6, risk_text)
    pdf.ln(4)

    # Probabilidades
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. Probabilidades por etapa", ln=True)
    pdf.set_font("Arial", "", 11)
    p_cn, p_emci, p_mci, p_ad = proba
    pdf.multi_cell(0, 6, "CN   : {:.3f}".format(p_cn))
    pdf.multi_cell(0, 6, "EMCI : {:.3f}".format(p_emci))
    pdf.multi_cell(0, 6, "MCI  : {:.3f}".format(p_mci))
    pdf.multi_cell(0, 6, "AD   : {:.3f}".format(p_ad))
    pdf.ln(4)

    # Biomarcadores
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "4. Biomarcadores del modelo", ln=True)
    pdf.set_font("Arial", "", 11)

    for k, v in feature_dict.items():
        pdf.multi_cell(0, 6, "{}: {:.4f}".format(k, v))

    pdf.ln(4)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0,
        5,
        "Este informe se basa en datos simulados y un modelo de investigacion. "
        "No constituye diagnostico clinico.",
    )

    # Export to bytes (evita errores de Unicode con 'replace')
    pdf_bytes = pdf.output(dest="S").encode("latin1", "replace")
    return pdf_bytes


# =========================================================
# 3. INTERFAZ STREAMLIT
# =========================================================

def main():
    st.set_page_config(
        page_title="Simulador de progresion de Alzheimer",
        layout="centered",
    )

    # Estilo global minimalista y formal (paleta azul, tipografía serif)
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #f0f4f8;
            font-family: "Georgia", "Times New Roman", serif;
        }
        [data-testid="stHeader"] {
            background-color: #f0f4f8;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: "Georgia", "Times New Roman", serif;
            color: #1a365d;
        }
        .stButton>button {
            background-color: #1a4b84;
            color: #ffffff;
            border-radius: 4px;
            border: 1px solid #1a4b84;
            padding: 0.4rem 0.75rem;
            font-family: "Georgia", "Times New Roman", serif;
        }
        .stButton>button:hover {
            background-color: #16355c;
            border-color: #16355c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Simulador de biomarcadores y riesgo de Alzheimer")

    st.markdown(
        """
Esta aplicación permite explorar, con fines estrictamente académicos,
cómo los biomarcadores simulados (conectomas individuales, modelo de Wilson–Cowan
y señales M/EEG sintéticas) se traducen en un índice ordinal de severidad
(0 = CN, 3 = AD).

Los resultados no constituyen un diagnóstico clínico y deben interpretarse
siempre en el contexto de la valoración médica integral.
"""
    )

    # Cargar modelo
    try:
        mlp, scaler = load_model_and_scaler()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Estado de la "escena" / paso
    if "step" not in st.session_state:
        st.session_state.step = 1

    if "result" not in st.session_state:
        st.session_state.result = None

    st.markdown("---")

    # =====================================================
    # PASO 1: DATOS DEL PACIENTE
    # =====================================================
    st.header("1. Datos del paciente")

    col1, col2 = st.columns(2)

    with col1:
        patient_id = st.text_input("Código del paciente", value="PX001")
        patient_name = st.text_input("Nombre", value="Paciente de ejemplo")
        sex = st.selectbox("Sexo", ["No especificado", "Masculino", "Femenino"])
        age = st.number_input("Edad (años)", min_value=0, max_value=120, value=70)

    with col2:
        schooling = st.number_input(
            "Escolaridad (años)", min_value=0, max_value=30, value=12
        )
        mmse = st.number_input(
            "MMSE (si se conoce)", min_value=0.0, max_value=30.0, value=26.0, step=0.5
        )
        cdr = st.number_input(
            "CDR (si se conoce)", min_value=0.0, max_value=3.0, value=0.5, step=0.5
        )

    notes = st.text_area("Notas clínicas / observaciones", height=80)

    # Botón para avanzar a Paso 2
    if st.button("Guardar datos y continuar a biomarcadores"):
        st.session_state.step = max(st.session_state.step, 2)

    st.markdown("---")

    # Empaquetar paciente (se usará luego)
    patient_info = {
        "id": patient_id,
        "name": patient_name,
        "sex": sex,
        "age": age,
        "schooling": schooling,
        "mmse": mmse,
        "cdr": cdr,
        "notes": notes,
    }

    # =====================================================
    # PASO 2: BIOMARCADORES DEL MODELO
    # =====================================================
    if st.session_state.step >= 2:
        st.header("2. Biomarcadores del modelo")

        st.markdown(
            """
Introduzca los **16 biomarcadores numéricos** derivados del pipeline:

- **P_E, G, noise_std, best_error** (parámetros y calidad de ajuste del modelo)
- **BP_delta, BP_theta, BP_alpha, BP_beta, BP_gamma** (bandpower normalizado)
- **theta_alpha_ratio, slowing** (índices espectrales derivados)
- **FC_strength, FC_clustering, FC_efficiency** (estadísticos de conectividad funcional)
- **SC_strength, SC_density** (estadísticos de conectividad estructural)
"""
        )

        colA, colB = st.columns(2)
        biomarker_values = {}

        with colA:
            biomarker_values["P_E"] = st.number_input("P_E", value=1.0, step=0.05)
            biomarker_values["G"] = st.number_input("G", value=0.6, step=0.05)
            biomarker_values["noise_std"] = st.number_input(
                "noise_std", value=0.2, step=0.05
            )
            biomarker_values["best_error"] = st.number_input(
                "best_error", value=0.10, step=0.01, format="%.4f"
            )

            biomarker_values["BP_delta"] = st.number_input(
                "BP_delta", min_value=0.0, max_value=1.0, value=0.30, step=0.01
            )
            biomarker_values["BP_theta"] = st.number_input(
                "BP_theta", min_value=0.0, max_value=1.0, value=0.31, step=0.01
            )
            biomarker_values["BP_alpha"] = st.number_input(
                "BP_alpha", min_value=0.0, max_value=1.0, value=0.19, step=0.01
            )
            biomarker_values["BP_beta"] = st.number_input(
                "BP_beta", min_value=0.0, max_value=1.0, value=0.17, step=0.01
            )

        with colB:
            biomarker_values["BP_gamma"] = st.number_input(
                "BP_gamma", min_value=0.0, max_value=1.0, value=0.03, step=0.01
            )
            biomarker_values["theta_alpha_ratio"] = st.number_input(
                "theta_alpha_ratio", value=1.5, step=0.1
            )
            biomarker_values["slowing"] = st.number_input(
                "slowing", value=0.5, step=0.05
            )

            biomarker_values["FC_strength"] = st.number_input(
                "FC_strength", value=0.1, step=0.01
            )
            biomarker_values["FC_clustering"] = st.number_input(
                "FC_clustering", value=0.2, step=0.01
            )
            biomarker_values["FC_efficiency"] = st.number_input(
                "FC_efficiency", value=0.3, step=0.01
            )

            biomarker_values["SC_strength"] = st.number_input(
                "SC_strength", value=0.1, step=0.01
            )
            biomarker_values["SC_density"] = st.number_input(
                "SC_density", value=0.2, step=0.01
            )

        # Botón para calcular modelo
        if st.button("Calcular riesgo y ver resultados"):
            # Construir vector en el orden correcto
            x_vec = np.array(
                [biomarker_values[name] for name in FEATURE_NAMES], dtype=float
            ).reshape(1, -1)

            # Verificar compatibilidad con scaler
            try:
                if x_vec.shape[1] != scaler.n_features_in_:
                    st.error(
                        f"Número de features = {x_vec.shape[1]}, "
                        f"pero el scaler espera {scaler.n_features_in_}. "
                        "Revise el orden o la lista de biomarcadores."
                    )
                    return
            except AttributeError:
                pass

            X_scaled = scaler.transform(x_vec)
            proba = mlp.predict_proba(X_scaled)[0]  # [p_CN, p_EMCI, p_MCI, p_AD]
            y_pred = int(mlp.predict(X_scaled)[0])
            stage_pred = STAGE_LABELS.get(y_pred, f"Clase {y_pred}")
            severity_soft = float(np.sum(proba * np.array([0, 1, 2, 3])))

            risk_zone, risk_text = compute_risk_zone(proba, severity_soft)

            # Guardar en session_state
            st.session_state.result = {
                "biomarkers": biomarker_values,
                "proba": proba,
                "y_pred": y_pred,
                "stage_pred": stage_pred,
                "severity_soft": severity_soft,
                "risk_zone": risk_zone,
                "risk_text": risk_text,
                "patient_info": patient_info,
            }
            st.session_state.step = 3

        st.markdown("---")

    # =====================================================
    # PASO 3: RESULTADOS Y TABS
    # =====================================================
    if st.session_state.step >= 3 and st.session_state.result is not None:
        res = st.session_state.result
        biomarker_values = res["biomarkers"]
        proba = res["proba"]
        y_pred = res["y_pred"]
        stage_pred = res["stage_pred"]
        severity_soft = res["severity_soft"]
        risk_zone = res["risk_zone"]
        risk_text = res["risk_text"]
        patient_info = res["patient_info"]

        st.header("3. Resultados del modelo y alerta clínica")

        # Tabs
        tab_summary, tab_spectrum, tab_probs, tab_details = st.tabs(
            [
                "Resumen ejecutivo",
                "Perfil espectral (comparativa)",
                "Probabilidades por etapa",
                "Detalles numéricos",
            ]
        )

        # ----------------------- TAB 1: RESUMEN -----------------------
        with tab_summary:
            # Colores por zona (paleta suave)
            if risk_zone == "Zona verde":
                bg_color = "#e2f3e8"
                border_color = "#8bbf9f"
            elif risk_zone == "Zona amarilla":
                bg_color = "#fff4d6"
                border_color = "#f2c97d"
            else:  # Zona roja
                bg_color = "#fde7e9"
                border_color = "#e39ba4"

            st.markdown(
                f"""
                <div style="
                    padding:14px;
                    border-radius:8px;
                    background-color:{bg_color};
                    border:1px solid {border_color};
                    font-size:15px;
                ">
                <strong>Estado de alerta:</strong> {risk_zone}. {risk_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")

            # Métricas clave
            st.subheader("Métricas clave")

            colM1, colM2, colM3 = st.columns(3)

            with colM1:
                st.markdown("**Etapa estimada**")
                st.markdown(f"### {stage_pred}")
                p_stage = float(proba[y_pred])
                st.markdown(
                    f"Probabilidad asociada: {p_stage*100:.1f} %"
                )

            with colM2:
                st.markdown("**Índice de severidad**")
                st.markdown(f"### {severity_soft:.2f}")

            with colM3:
                st.markdown("**Número de biomarcadores**")
                st.markdown(f"### {len(FEATURE_NAMES)}")

            st.markdown("---")

            st.markdown(
                "**Interpretación sugerida (no clínica):** "
                "El patrón observado es compatible con el fenotipo **{}**, de acuerdo con los biomarcadores "
                "simulados (dinámica espectral y conectividad estructural/funcional). "
                "Se recomienda interpretar estos resultados junto con la historia clínica, "
                "las escalas MMSE/CDR y otros estudios complementarios.".format(stage_pred)
            )

        # ----------------------- TAB 2: PERFIL ESPECTRAL ----------------
        with tab_spectrum:
            st.subheader("Distribución espectral normalizada")

            bands = ["delta", "theta", "alpha", "beta", "gamma"]
            bp_values = np.array(
                [
                    biomarker_values["BP_delta"],
                    biomarker_values["BP_theta"],
                    biomarker_values["BP_alpha"],
                    biomarker_values["BP_beta"],
                    biomarker_values["BP_gamma"],
                ],
                dtype=float,
            )
            total_bp = float(bp_values.sum())
            if total_bp > 0:
                bp_norm = bp_values / total_bp
            else:
                bp_norm = np.zeros_like(bp_values)

            fig1, ax1 = plt.subplots()
            ax1.bar(bands, bp_norm, color="#4a5568", edgecolor="#2d3748")
            ax1.set_ylabel("Fracción de potencia (normalizada)")
            ax1.set_title("Perfil espectral del sujeto")
            ax1.set_ylim(0, max(0.4, float(bp_norm.max()) + 0.05))
            ax1.grid(axis="y", linestyle="--", linewidth=0.5, color="#cbd5e0", alpha=0.7)
            fig1.set_facecolor("#ffffff")
            ax1.set_facecolor("#ffffff")
            for spine in ax1.spines.values():
                spine.set_color("#a0aec0")
            st.pyplot(fig1)

            st.markdown(
                f"""
**Valores normalizados aproximados:**

- Delta: **{bp_norm[0]:.2f}**
- Theta: **{bp_norm[1]:.2f}**
- Alpha: **{bp_norm[2]:.2f}**
- Beta : **{bp_norm[3]:.2f}**
- Gamma: **{bp_norm[4]:.2f}**

Un aumento de la potencia relativa en bandas **delta/theta** junto con una reducción
en **alpha/beta** se asocia con el *slowing* típico descrito en la progresión
del Alzheimer.
"""
            )

        # ----------------------- TAB 3: PROBABILIDADES ------------------
        with tab_probs:
            st.subheader("Distribución de probabilidad por etapa")

            stage_names = ["CN", "EMCI", "MCI", "AD"]
            fig2, ax2 = plt.subplots()
            ax2.bar(stage_names, proba, color="#4a5568", edgecolor="#2d3748")
            ax2.set_ylabel("Probabilidad")
            ax2.set_ylim([0, 1])
            ax2.set_title("Probabilidades estimadas (salida del modelo)")
            ax2.grid(axis="y", linestyle="--", linewidth=0.5, color="#cbd5e0", alpha=0.7)
            fig2.set_facecolor("#ffffff")
            ax2.set_facecolor("#ffffff")
            for spine in ax2.spines.values():
                spine.set_color("#a0aec0")
            st.pyplot(fig2)

            st.write(
                f"""
**Valores numéricos:**

- p(CN)   = {proba[0]:.3f}  
- p(EMCI) = {proba[1]:.3f}  
- p(MCI)  = {proba[2]:.3f}  
- p(AD)   = {proba[3]:.3f}  

La etapa estimada corresponde a la clase con **probabilidad máxima**, en este caso: **{stage_pred}**.
"""
            )

        # ----------------------- TAB 4: DETALLES NUMÉRICOS -------------
        with tab_details:
            st.subheader("Biomarcadores usados por el modelo")

            feature_table = {
                "Biomarcador": [],
                "Valor": [],
            }
            for k in FEATURE_NAMES:
                feature_table["Biomarcador"].append(k)
                feature_table["Valor"].append(biomarker_values[k])

            df_features = pd.DataFrame(feature_table)
            st.dataframe(df_features, use_container_width=True)

        # =====================================================
        # 4. DESCARGA DE INFORME EN PDF
        # =====================================================
        st.markdown("---")
        st.header("4. Descarga de informe en PDF")

        if not FPDF_AVAILABLE:
            st.warning(
                "La librería `fpdf` no está instalada. "
                "Instálela con `pip install fpdf` para habilitar la descarga en PDF."
            )
        else:
            pdf_bytes = build_pdf_report(
                patient_info=patient_info,
                feature_dict={k: biomarker_values[k] for k in FEATURE_NAMES},
                stage_pred=stage_pred,
                severity_soft=severity_soft,
                proba=proba,
                risk_zone=risk_zone,
                risk_text=risk_text,
            )

            st.download_button(
                label="Descargar informe en PDF",
                data=pdf_bytes,
                file_name=f"informe_{patient_id}.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
