import os
import warnings
from datetime import timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------
# Page & global settings
# ---------------------------
st.set_page_config(page_title="Debris Bed AI", layout="wide")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# ---------------------------
# Geometry utilities
# ---------------------------
def compute_geometry(Mfuel, Ffuel, Fstruct, Porosity, Rflat, Alpha):
    Pi = 4.0 * np.arctan(1.0)
    Mcorium = Mfuel * Ffuel * (1 + Fstruct)
    Vdebris = Mcorium / 7300.0 / (1.0 - Porosity)

    Rcav, Hcav, Hwater = 4.4, 5.0, 5.0
    Rlow, Hlow = 3.35, 1.0

    alpha = Alpha * Pi / 180.0
    tana = np.tan(alpha)

    V1 = Pi / 3.0 * tana * (Rlow**3 - Rflat**3)
    V2 = V1 + Pi * Hlow * Rlow**2
    V3 = V2 - V1 + Pi / 3.0 * tana * (Rcav**3 - Rflat**3)

    if Vdebris < V1:
        R = (Rflat**3 + 3.0 * Vdebris / (Pi * tana)) ** (1.0 / 3.0)
        H = tana * (R - Rflat)
        rDebris = [0.0, R, Rflat, 0.0]
        zDebris = [0.0, 0.0, H, H]
    elif Vdebris < V2:
        Vcyl = Vdebris - V1
        Hcyl = Vcyl / (Pi * Rlow**2)
        Hcon = tana * (Rlow - Rflat)
        H = Hcyl + Hcon
        rDebris = [0.0, Rlow, Rlow, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hcyl, H, H]
    elif Vdebris < V3:
        Vcon = Vdebris - Pi * Hlow * Rlow**2
        R = (Rflat**3 + 3.0 * Vcon / (Pi * tana)) ** (1.0 / 3.0)
        Hcon = tana * (R - Rflat)
        H = Hlow + Hcon
        rDebris = [0.0, Rlow, Rlow, R, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hlow, Hlow, H, H]
    else:
        Vcyl = Vdebris - V3
        Hcyl = Vcyl / (Pi * Rcav**2) + Hlow
        Hcon = tana * (Rcav - Rflat)
        H = Hcyl + Hcon
        rDebris = [0.0, Rlow, Rlow, Rcav, Rcav, Rflat, 0.0]
        zDebris = [0.0, 0.0, Hlow, Hlow, Hcyl, H, H]

    return rDebris, zDebris


def plot_geometry(rDebris, zDebris):
    fig, ax = plt.subplots(figsize=(4, 4))

    # Background water
    ax.set_facecolor("lightblue")
    ax.fill_betweenx([0, 5], 0, 4.4, color="lightblue", alpha=0.5, label="Wasser")

    # Debris shape
    ax.fill(rDebris, zDebris, facecolor="red", alpha=0.5, edgecolor="black", label="Schüttbett")

    # Block region (between r = 3.35 and r = 4.4, 0 <= z <= 1)
    block_start = 3.35
    block_end = 4.4
    ax.fill_betweenx([0, 1], block_start, block_end, color="grey", alpha=0.9, label="Block")

    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 5)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    return fig


# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models(max_models: int = 6):
    models = []
    for i in range(max_models):
        path = f"model_{i}.keras"
        if os.path.exists(path):
            try:
                models.append(tf.keras.models.load_model(path))
            except Exception as e:
                st.error(f"Failed to load {path}: {e}")
    return models


@st.cache_resource(show_spinner=False)
def load_classifier(path: str = "OptimizedVotingClassifier.pkl"):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_scaler_X(path: str = "MinMax_scaler_X_Classification.pkl"):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load X-scaler: {e}")
        return None


# Simple wrappers for target scaling (no caching needed)
def forward_y(v):
    return np.log1p(v)


def inverse_y(v):
    return np.expm1(v)


# ---------------------------
# Load artifacts
# ---------------------------
models = load_models()
classifier = load_classifier()
scaler_X = load_scaler_X()

# Guardrails
if scaler_X is None:
    st.stop()

# Use feature names from the scaler (drives UI order)
param_names = list(getattr(scaler_X, "feature_names_in_", []))
if not param_names:
    # Fallback: hardcode names in the correct training order
    param_names = [
        "p_sys_MPa",
        "F_fuel",
        "porosity",
        "d_mean_mm",
        "alpha_deg",
        "R_flat_m",
        "T0_K",
        "Q_decay_rel",
        "F_struct",
    ]

# Descriptions & slider limits MUST match param_names order
labels = [
    "System pressure [MPa]",
    "Ratio of fuel relocated to total fuel [-]",
    "Porosity of the packed bed [-]",
    "Mean particle diameter [mm]",
    "Angle of repose of the packed bed [°]",
    "Upper radius of the truncated cone [m]",
    "Initial temperature of the dry packed bed [K]",
    "Decay heat / nominal power [%]",
    "Relocated cladding+structure / relocated fuel [-]",
]

min_vals = [0.11, 0.5, 0.25, 1.0, 15.0, 0.0, 400.0, 0.3, 0.25]
max_vals = [0.5, 1.0, 0.5, 5.0, 45.0, 2.0, 1700.0, 1.0, 2.0]

# ---------------------------
# Header
# ---------------------------
st.markdown("---")
st.title("Debris Bed AI: Prediction of Quench Behavior")

st.markdown(
    """
### ℹ️ Instructions

Use the sliders on the left to adjust the parameters of the debris bed.  
The geometry updates on the right.  
If cooling is possible, the app predicts the quench time using an ensemble of AI models.
"""
)

with st.expander("🛠️ More Help & Background Information"):
    st.markdown(
        """
- Predictions are based on an ensemble of neural networks.
- Inputs are normalized using the same `MinMaxScaler` that was used for training.
- If quench/remelt is predicted by the classifier, the NN ensemble estimates the time.
- Uncertainty is the standard deviation across the ensemble.
- This demo is **not** a safety-relevant assessment.
"""
    )

st.markdown("---")

# ---------------------------
# Layout
# ---------------------------
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### Input parameters of the debris bed:")
    st.markdown("---")

    raw_values = []
    for i in range(len(param_names)):
        val = st.slider(
            label=labels[i],
            min_value=float(min_vals[i]),
            max_value=float(max_vals[i]),
            value=float((min_vals[i] + max_vals[i]) / 2.0),
            key=f"slider_{i}",
            format="%.2g",
        )
        raw_values.append(val)

    # Scale with the **trained scaler** to match the classifier/NNs
    X_scaled = scaler_X.transform(np.array(raw_values, dtype=float).reshape(1, -1))[0]

with col2:
    # Geometry inputs (from sliders)
    Ffuel = float(st.session_state.get("slider_1", 0.75))
    Porosity = float(st.session_state.get("slider_2", 0.35))
    Alpha = float(st.session_state.get("slider_4", 30.0))
    Rflat = float(st.session_state.get("slider_5", 1.0))
    Fstruct = float(st.session_state.get("slider_8", 1.0))

    Mfuel = 136000.0  # kg
    rDebris, zDebris = compute_geometry(Mfuel, Ffuel, Fstruct, Porosity, Rflat, Alpha)

    st.markdown("### Geometry of the Debris Bed:")
    st.pyplot(plot_geometry(rDebris, zDebris))
    st.markdown("---")

    # ---------------------------
    # Classification
    # ---------------------------
    if classifier is None:
        st.warning("Classifier not available – cannot make predictions.")
    else:
        try:
            prediction = int(classifier.predict([X_scaled])[0])
        except Exception as e:
            st.error(f"Classification failed: {e}")
            prediction = None

        if prediction == 1:
            st.markdown("### ✅ Debris bed quenches:")
        elif prediction == 0:
            st.markdown("### ❌ Debris bed will remelt:")
        elif prediction == 2:
            st.markdown("### ⏳ Inconclusive: No definitive answer within 2 hours.")

        # ---------------------------
        # Ensemble regression (if we have models)
        # ---------------------------
        if prediction in (0, 1):
            if not models:
                st.info("No Keras models found (model_*.keras). Skipping time prediction.")
            else:
                # Augment input with class flag as in training
                cls = 1 if prediction == 1 else 0
                x = np.append(X_scaled.reshape(1, -1), [[cls]], axis=1)
                # add error messge to check shape and input of x
                st.write(f"Input shape to models: {x.shape}")
                st.write(f"Input values to models: {x}")

                preds = []
                for i, m in enumerate(models):
                    try:
                        yhat = m.predict(x, verbose=0)[0][0]
                        preds.append(float(yhat))
                    except Exception as e:
                        st.warning(f"Model {i+1} failed: {e}")

                if preds:
                    avg = float(np.mean(preds))
                    std = float(np.std(preds))

                    # Inverse transform
                    avg_s = inverse_y(avg)
                    std_s = inverse_y(std)

                    def to_hms(seconds: float) -> str:
                        seconds = max(0, float(seconds))
                        total = int(round(seconds))
                        h, rem = divmod(total, 3600)
                        m, s = divmod(rem, 60)
                        return f"{h}h {m}m {s}s"

                    st.markdown(f"## {'Quenching' if cls==1 else 'Time until melting'}: {to_hms(avg_s)}")
                    st.markdown(f"### Uncertainty: {to_hms(std_s)}")

                    # Static plot of individual predictions (in seconds)
                    preds_s = inverse_y(np.array(preds).reshape(-1, 1)).flatten()

                    def plot_ensemble_non_interactive(values):
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(values, marker="o", label="Prediction")
                        ax.set_title("Ensemble Prediction [s]")
                        ax.set_ylabel("End time [s]")
                        ax.set_xticks(np.arange(len(values)))
                        ax.set_xticklabels([f"Model {i+1}" for i in range(len(values))])
                        ax.legend(loc="best")
                        ax.grid(True)
                        return fig

                    st.pyplot(plot_ensemble_non_interactive(preds_s))

                    with st.expander("What is Uncertainty?"):
                        st.markdown(
                            """
The uncertainty reflects:
- **Model variation** across ensemble members,
- **Data uncertainty** in inputs,
- **Stochastic effects** of the system.

It is computed as the **standard deviation** of the ensemble predictions.
"""
                        )

                    with st.expander("What is an Ensemble?"):
                        st.markdown(
                            """
An ensemble combines several models to stabilize and improve predictions.  
Variation among members is used to estimate confidence in the result.
"""
                        )

# ---------------------------
# Footer
# ---------------------------
# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em;'>
    <strong>Developed by:</strong> Jasmin Joshi-Thompson, University of Stuttgart, Institute for Nuclear Energy and Energy Systems (IKE), 2025 <br>
    <strong>Contact:</strong> Jasmin.Joshi-Thompson@ike.uni-stuttgart.de <br>
    <strong>Version:</strong> 2.0 – Last updated: August 2025<br><br>
    <em>This app uses AI models for live prediction of quench and melting times.</em><br>
</div>
""", unsafe_allow_html=True)

# Add the expander for the reference information
with st.expander("Note"):
    st.markdown("""
    This AI model was pre-trained with simulation data from **COCOMO** (Corium Coolability Model),  
    based on work published at the **NENE Conference 2025**, following the work from **NENE 2024** [1].  
    The simulation data was validated against experimental data from the **FLOAT test facility** [2].  
    COCOMO was developed at the **Institute for Nuclear Energy and Energy Systems (IKE)** at the **University of Stuttgart** [3].
    
    **References:**  
    [1] Joshi-Thompson, J., Buck, M., and Starflinger, J., "Application of AI Methods for Describing the Coolability of Debris Beds Formed in the Late Accident Phase of Nuclear Reactors",  
    *Proceedings of the 33rd International Conference Nuclear Energy for New Europe (NENE 2024)*, Portorož, Slovenia, September 9–12, 2024.

    [2] M. Petroff, R. Kulenovic, and J. Starflinger, "Experimental investigation on debris bed quenching with additional non-condensable gas injection",  
    *Journal of Nuclear Engineering and Radiation Science*, NERS-21-1028, 2022.

    [3] Buck, M., and Pohlner, G., "Ex-Vessel Debris Bed Formation and Coolability – Challenges and Chances for Severe Accident Mitigation",  
    *Proceedings of the International Congress on Advances in Nuclear Power Plants (ICAPP 2016)*, San Francisco, USA, April 17–20, 2016.
    """, unsafe_allow_html=True)

