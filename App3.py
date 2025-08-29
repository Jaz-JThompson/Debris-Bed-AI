import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
from datetime import timedelta
import matplotlib.pyplot as plt

# --- Settings ---
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
tf.compat.v1.reset_default_graph()

# --- Utility functions ---
def compute_geometry(Mfuel, Ffuel, Fstruct, Porosity, Rflat, Alpha):
    Pi = np.pi
    Mcorium = Mfuel * Ffuel * (1 + Fstruct)
    Vdebris = Mcorium / 7300.0 / (1.0 - Porosity)
    Rcav, Hcav, Hwater = 4.4, 5.0, 5.0
    Rlow, Hlow = 3.35, 1.0
    alpha = np.radians(Alpha)
    tana = np.tan(alpha)

    V1 = Pi / 3.0 * tana * (Rlow**3 - Rflat**3)
    V2 = V1 + Pi * Hlow * Rlow**2
    V3 = V2 - V1 + Pi / 3.0 * tana * (Rcav**3 - Rflat**3)

    if Vdebris < V1:
        R = (Rflat**3 + 3.0 * Vdebris / (Pi * tana))**(1.0/3.0)
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
        R = (Rflat**3 + 3.0 * Vcon / (Pi * tana))**(1.0/3.0)
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
    ax.set_facecolor('lightblue')
    ax.fill_betweenx([0, 5], 0, 4.4, color='lightblue', alpha=0.5, label='Wasser')
    ax.fill(rDebris, zDebris, facecolor='red', alpha=0.5, edgecolor='black', label='Schüttbett')
    ax.fill_betweenx([0, 1], 3.35, 4.4, color='grey', alpha=0.9, label='Block')
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 5)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    return fig


def scaler_y(value):
    return np.log1p(value)


def inverse_scaler_y(value):
    return np.expm1(value)

# --- Resource loaders ---
@st.cache_resource
def load_models():
    models = []
    for i in range(5):
        path = f"model_{i}.keras"
        if os.path.exists(path):
            try:
                models.append(tf.keras.models.load_model(path))
            except Exception as e:
                st.error(f"Error loading {path}: {e}")
    return models

@st.cache_resource
def load_classifier():
    try:
        return joblib.load("OptimizedVotingClassifier.pkl")
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None

@st.cache_resource
def load_scaler_X():
    try:
        return joblib.load("MinMax_scaler_X_Classification.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# --- Load resources ---
models = load_models()
classifier = load_classifier()
scaler_X = load_scaler_X()

# --- UI ---
st.title("Debris Bed AI: Prediction of Quench Behavior")

col1, col2 = st.columns([2, 3])

# Descriptions and reasonable UI ranges (fallback)
descriptions = [
    "System pressure [MPa]",
    "Ratio of relocated fuel [-]",
    "Porosity [-]",
    "Mean particle diameter [mm]",
    "Angle of repose [°]",
    "Upper radius of truncated cone [m]",
    "Initial temperature [K]",
    "Decay heat ratio [%]",
    "Cladding/fuel ratio [-]"
]
min_vals = [0.11, 0.5, 0.25, 1.0, 15.0, 0.0, 400.0, 0.3, 0.25]
max_vals = [0.5, 1.0, 0.5, 5.0, 45.0, 2.0, 1700.0, 1.0, 2.0]

# Determine slider keys/order: prefer scaler.feature_names_in_ when available
if scaler_X is not None and hasattr(scaler_X, 'feature_names_in_'):
    feature_order = list(scaler_X.feature_names_in_)
else:
    # fallback to a generic order matching the descriptions
    feature_order = [f"f{i}" for i in range(len(descriptions))]

# Build a label map for display: try to map feature names to descriptions if lengths match
if len(feature_order) == len(descriptions):
    label_map = dict(zip(feature_order, descriptions))
    min_map = dict(zip(feature_order, min_vals))
    max_map = dict(zip(feature_order, max_vals))
else:
    label_map = {name: name for name in feature_order}
    min_map = {name: 0.0 for name in feature_order}
    max_map = {name: 1.0 for name in feature_order}

# Render sliders and store values in session_state keys that match feature_order
with col1:
    st.subheader("Input Parameters")
    for name in feature_order:
        vmin = float(min_map.get(name, 0.0))
        vmax = float(max_map.get(name, 1.0))
        default = float((vmin + vmax) / 2.0)
        # Use the feature name as the key to ensure consistent ordering
        st.slider(label_map.get(name, name), min_value=vmin, max_value=vmax, value=default, key=name, format="%.4g")

# Collect raw inputs in the exact order of the scaler
raw_inputs = np.array([st.session_state.get(name) for name in feature_order], dtype=float)
raw_inputs[0] = raw_inputs[0] * 1e6 # MPa -> Pa
# Debug: show raw inputs
st.write("Raw input values (in UI order):", raw_inputs)

# Clip inputs to scaler's data range to avoid negative scaled values when input < fitted min
if scaler_X is not None and hasattr(scaler_X, 'data_min_') and hasattr(scaler_X, 'data_max_'):
    data_min = scaler_X.data_min_
    data_max = scaler_X.data_max_
    clipped = np.minimum(np.maximum(raw_inputs, data_min), data_max)
    if not np.allclose(clipped, raw_inputs):
        st.warning("Some inputs were clipped to the scaler's fitted min/max to avoid out-of-range scaling.")
        st.write("Clipped inputs:", clipped)
    processed_inputs = clipped
else:
    processed_inputs = raw_inputs

# Transform using scaler (safe)
if scaler_X is not None:
    try:
        user_inputs_scaled = scaler_X.transform([processed_inputs])
        st.write("Scaled input (after clipping/ordering):", user_inputs_scaled)
    except Exception as e:
        st.error(f"Error scaling input: {e}")
        user_inputs_scaled = None
else:
    st.error("Scaler not available. Cannot scale inputs.")
    user_inputs_scaled = None

# --- Geometry plot (uses original physical parameter order from UI - attempt to map sensible indices) ---
try:
    # Attempt to find the keys we need in the feature_order; fallback to positions if keys are generic
    def get_by_name_or_idx(name, idx):
        if name in feature_order:
            return st.session_state[name]
        else:
            return raw_inputs[idx]

    Ffuel = get_by_name_or_idx('Ffuel', 1)  # example feature name candidate
    Porosity = get_by_name_or_idx('Porosity', 2)
    Alpha = get_by_name_or_idx('Alpha', 4)
    Rflat = get_by_name_or_idx('Rflat', 5)
    Fstruct = get_by_name_or_idx('Fstruct', 8)
except Exception:
    # Generic fallback if feature names are unknown
    Ffuel = raw_inputs[1] if len(raw_inputs) > 1 else 0.6
    Porosity = raw_inputs[2] if len(raw_inputs) > 2 else 0.3
    Alpha = raw_inputs[4] if len(raw_inputs) > 4 else 30.0
    Rflat = raw_inputs[5] if len(raw_inputs) > 5 else 0.5
    Fstruct = raw_inputs[8] if len(raw_inputs) > 8 else 0.25

rDebris, zDebris = compute_geometry(136000.0, Ffuel, Fstruct, Porosity, Rflat, Alpha)
with col2:
    st.subheader("Geometry of the Debris Bed")
    st.pyplot(plot_geometry(rDebris, zDebris))

# --- Classifier & regressor pipeline ---
if classifier is None:
    st.error("Classifier is not loaded. Check file OptimizedVotingClassifier.pkl")
elif user_inputs_scaled is None:
    st.error("Scaled inputs missing. Cannot run classifier.")
else:
    try:
        prediction = classifier.predict(user_inputs_scaled)[0]
        st.write("Classifier prediction (0=remelt,1=quench,...):", prediction)

        if prediction in (0, 1):
            if not models:
                st.info("No regression models found (model_*.keras). Skipping time prediction.")
            else:
                # Create class one-hot consistent with training: search classifier classes if available
                if hasattr(classifier, 'classes_'):
                    # if classes are [0,1] typical
                    cls = [0, 1] if prediction == 1 else [1, 0]
                else:
                    cls = [0, 1] if prediction == 1 else [1, 0]

                input_array = np.concatenate([user_inputs_scaled, np.array(cls).reshape(1, -1)], axis=1)
                st.write("Regressor input array (scaled + class):", input_array)

                preds = []
                for m in models:
                    try:
                        p = m.predict(input_array)[0][0]
                        preds.append(p)
                    except Exception as e:
                        st.warning(f"Model prediction failed for one model: {e}")

                st.write("Raw regressor outputs (ensemble):", preds)

                if preds:
                    avg = np.mean(preds)
                    std = np.std(preds)

                    # Inverse transform (safely) and format
                    inv_avg = inverse_scaler_y(np.array([[avg]])).item()
                    inv_std = inverse_scaler_y(np.array([[std]])).item()

                    def format_td(seconds):
                        try:
                            s = float(seconds)
                            td = timedelta(seconds=s)
                            total_seconds = int(td.total_seconds())
                            hours, remainder = divmod(total_seconds, 3600)
                            minutes, seconds = divmod(remainder, 60)
                            return f"{hours}h {minutes}m {seconds}s"
                        except Exception:
                            return str(seconds)

                    st.subheader("Predicted Quench Time")
                    st.markdown(f"**Average:** {format_td(inv_avg)}")
                    st.markdown(f"**Uncertainty:** {format_td(inv_std)}")

    except Exception as e:
        st.error(f"Error in prediction pipeline: {e}")

# Footer
st.markdown("---")
st.caption("Debugging: check the 'Raw input values' and 'Scaled input' prints above. If any scaled value is negative, ensure the slider lower bounds are not below the scaler's fitted minimum or let the app clip inputs.")
