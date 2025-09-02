import streamlit as st
import numpy as np
import tensorflow as tf
tf.compat.v1.reset_default_graph() 
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
from datetime import timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import keras
st.write("TensorFlow version:", tf.__version__)
st.write("Keras version:", keras.__version__)


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
    
    # Set the background color to blue
    ax.set_facecolor('lightblue')
    

    # Adding the "Wasser" background as a label in the legend
    ax.fill_betweenx([0, 5], 0, 4.4, color='lightblue', alpha=0.5, label='Water')
        # Plot the debris shape
    ax.fill(rDebris, zDebris, facecolor='red', alpha=0.5, edgecolor='black', label='Debris Bed')
    
    # Define the block region (example: between r = 1.0 and r = 2.0)
    block_start = 3.35  # ['3.35', ' 4.4', ' 4.4', ' 3.35']
    block_end = 4.4
    ax.fill_betweenx(
        [0, 1], block_start, block_end, color='grey', alpha=0.9, label='Structural Block'
    )

    
    # Set plot limits
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 5)
    
    # Labels and grid
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True)

    # Adding a legend
    ax.legend()

    return fig


# Suppress the feature name warning
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)  # directory where app2.py lives
    models = []
    for i in range(6):
        path = os.path.join(base_dir, f"model_{i}.keras")
        if os.path.exists(path):
            models.append(tf.keras.models.load_model(path, compile=False))
        else:
            st.warning(f"Model file not found: {path}")
    return models

def load_classifier():
    # Load the saved model
    return joblib.load("OptimizedVotingClassifier.pkl")


@st.cache_resource
def load_scaler_X():
    return joblib.load("MinMax_scaler_X_Classification.pkl")


def scaler_y(value):
    value =  np.log1p(value)
    return value

def inverse_scaler_y(value):
    value = np.expm1(value)
    return value

models = load_models()
scaler_X = load_scaler_X()


# Use feature names directly from the scaler
param_names = scaler_X.feature_names_in_.tolist()


# Manually define descriptions and limits
descriptions = [
    "System pressure [MPa]",
    "Ratio of the mass of fuel relocated from the reactor pressure vessel to the total mass of the fuel [-]",
    "Porosity of the packed bed [-]",
    "Mean particle diameter [mm]",
    "Angle of repose of the packed bed [°]",
    "Upper radius of the truncated cone [m]",
    "Initial temperature of the dry packed bed [K]",
    "Ratio of the decay heat to the thermal power during nominal operation [%]",
    "Ratio of the mass of relocated cladding and structural material to the mass of fuel relocated from the reactor pressure vessel [-]"
]

min_vals = [0.11, 0.5, 0.25, 1.0, 15.0, 0.0, 400.0, 0.3, 0.25]
max_vals = [0.5, 1.0, 0.5, 5.0, 45.0, 2.0, 1700.0, 1, 2.0]

#Hedder 
st.markdown("---")
# add picture
st.title("Debris Bed AI: Prediction of Quench Behavior")

st.markdown("""
### ℹ️ Instructions

Use the sliders on the left side to adjust the parameters of the debris bed.  
The geometry will automatically update and be displayed on the right.  
If a definitive conclusion can be reached, the application will calculate a live prediction of the quench or melting time using AI models.
""")

with st.expander("🛠️ More Help & Background Information"):
    st.markdown("""
- The predictions are based on an ensemble of multiple neural networks.
- Input values are automatically normalized before being fed into the AI model.
- If quenching or melting is predicted by the classifier, the NN is informed and the expected quench/melting time is calculated.
- The prediction uncertainty is derived from the variation across all model results.
- The visualization shows both the geometric arrangement of the debris bed and the prediction plot.
- The app was developed for demonstration purposes does not provide safety-relevant assessments.

For questions or feedback, please contact the development team.
""")
st.markdown("---")


# Setup layout with two columns
col1, col2 = st.columns([2, 3])  # Left column (1 part) and right column (2 parts)

# Left column for parameter sliders
with col1:
   

    # Build sliders with manually defined parameters
    user_inputs = []
    st.markdown("### Input parameters of the debris bed:")
    st.markdown("---")
    for i in range(len(param_names)):

        # Show slider in real-world units
        val = st.slider(
            label=f"{descriptions[i]}",  # (`{param_names[i]}`)",
            min_value=float(min_vals[i]),
            max_value=float(max_vals[i]),
            value=(min_vals[i] + max_vals[i]) / 2,
            key=f"slider_{i}",
            format="%.2g"  # 2 significant figures
        )

        user_inputs.append(val)

        
        #st.caption(f"Skalierter Wert: {scaled_val:.2f}")
#apply min-max scaling
#multiply the pressure by 1000000 to convert from MPa to Pa
user_inputs[0] = user_inputs[0] * 1e6  # Convert MPa to Pa
user_inputs[3] = user_inputs[3] / 1000.0  # Convert mm to m
user_inputs[7] = user_inputs[7] /100.0  # Convert percentage to fraction
user_inputs_scaled = scaler_X.transform([user_inputs])[0] #????? this fucked up the function -check it
# Right column for geometry plot and prediction output
with col2:
    # Compute geometry based on user inputs
    Ffuel = st.session_state["slider_1"]
    Porosity = st.session_state["slider_2"]
    Alpha = st.session_state["slider_4"]
    Rflat = st.session_state["slider_5"]
    Fstruct = st.session_state["slider_8"]

    Mfuel = 136000.0  # Fixed value for fuel mass, or add a slider if needed
    rDebris, zDebris = compute_geometry(Mfuel, Ffuel, Fstruct, Porosity, Rflat, Alpha)

    # Plot geometry
    fig = plot_geometry(rDebris, zDebris)
    st.markdown("### Geometry of the Debris Bed:")
    st.pyplot(fig)
    st.markdown("---")
    # ------------------------------------------------------------------------------
    # Load classifier model
    classifier = load_classifier()
    print(classifier.predict([user_inputs_scaled]))
    prediction = classifier.predict([user_inputs_scaled])[0]
    if prediction == 1:
        st.markdown("### ♨️ Debris bed quenches after:")
    elif prediction == 0:
        st.markdown("### 🌡️ Debris bed will remelt after:")
    elif prediction == 2:
        st.markdown("### 🤔 Inconclusive: A definitive answer cannot be reached after 2 hours")

    if prediction in (0, 1):
        if not models:
            st.info("No Keras models found (model_*.keras). Skipping time prediction.")
        else:
            # Augment input with class flag as in training
            cls = [0, 1] if prediction == 1 else [1, 0]

            # Input vorbereiten und Vorhersagen durchführen
            input_array = np.array(user_inputs_scaled).reshape(1, -1)

            # convert cls to shape (1,2) and concatenate
            cls_array = np.array(cls).reshape(1, -1)
            input_array = np.concatenate([input_array, cls_array], axis=1)
            #print the input shape and values


            predictions = [model.predict(input_array)[0][0] for model in models]
           
            avg = inverse_scaler_y(np.mean(predictions))
            
            print("Raw Predictions:", predictions)
            if predictions and len(predictions) > 0:
                avg = np.mean(predictions)
                scaled_avg = inverse_scaler_y([[avg]])[0][0]
                
                #scale each prediction to real units
                real_preds = inverse_scaler_y(np.array(predictions).reshape(-1, 1)).flatten()
                # Compute mean and standard deviation in real units
                avg_real = np.mean(real_preds)
                std_real = np.std(real_preds)
                # time format
                try:
                    predicted_duration = timedelta(seconds=float(scaled_avg))
                    uncertainty_duration = timedelta(seconds=float(std_real))

                    def format_timedelta(td):
                        total_seconds = int(td.total_seconds())
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        return f"{hours}h {minutes}m {seconds}s"

                    st.markdown(f"## {format_timedelta(predicted_duration)}")
                    #st.caption(f"Vorhersage der Quenchzeit: {scaled_avg:.2f} Sekunden")
                    st.markdown(f"### Uncertainty: {format_timedelta(uncertainty_duration)}")
                    with st.expander("What is Uncertainty?"):
                        st.markdown("""
                        The uncertainty in the predictions comes from several sources, including:
                        - **Model variations**: Different models may respond to the same input data in different ways.
                        - **Data uncertainty**: Variations and inaccuracies in the input data, such as measurement errors or incomplete data.
                        - **Stochastic processes**: Random fluctuations caused by the intrinsic nature of the system dynamics.
                        
                        The uncertainty is calculated using the **standard deviation** of the models' predictions.  
                        This indicates how much the models vary in their outputs and thus provides an estimate of the uncertainty of the ensemble prediction.
                        """)


                    # Plot erstellen
                    predictions = inverse_scaler_y(np.array(predictions).reshape(-1, 1)).flatten()
                    # Define x labels and numeric positions
                    model_names = [f"Model {i+1}" for i in range(len(predictions))] 
                    def plot_ensemble_non_interactive():
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # Plot the predictions
                        ax.plot(predictions, marker='o', label='Prediction', color='b')
                        
                        # Add labels and title
                        ax.set_title("Ensemble-Prediction [s]", fontsize=14)
                        #ax.set_xlabel("Modelle")
                        ax.set_ylabel("End time [s]")
                        ax.set_xticks(np.arange(len(model_names)))
                        ax.set_xticklabels(model_names)
                        
                        # Add a legend
                        ax.legend(loc='best')
                        
                        # Display a grid
                        ax.grid(True)

                        return fig

                    # Show the static plot in Streamlit
                    fig = plot_ensemble_non_interactive()
                    st.pyplot(fig)

                    # Help box with explanation
                    # Dropdown (Expander) for explanation with uncertainty and standard deviation
                    with st.expander("What is an Ensemble?"):
                        st.markdown("""
                        An ensemble combines the predictions of several models to stabilize and improve the overall prediction.  
                        This helps reduce errors from individual models and provides more robust and reliable results.
                        
                        An ensemble is also used to calculate how confident the model is in its prediction.  
                        This is done by analyzing the variation between the predictions of the individual models.  
                        Greater variation indicates higher uncertainty, while smaller variation indicates greater confidence in the model’s prediction.
                        """)
                except OverflowError:
                    st.error("The predicted quench time is too large to be represented.")
                    



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

