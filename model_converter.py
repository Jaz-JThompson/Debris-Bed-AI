import tensorflow as tf
from tensorflow import keras
import os

# List your original models
model_files = ["model_1.keras", "model_2.keras", "model_3.keras", "model_4.keras", "model_5.keras"]

# Create output directory for new models
os.makedirs("converted_models", exist_ok=True)

for f in model_files:
    # Load original model
    model = keras.models.load_model(f)
    
    # Create new filename
    name = os.path.splitext(os.path.basename(f))[0]
    new_name = f"model_new_{name}.keras"
    out_path = os.path.join("converted_models", new_name)
    
    # Save as new .keras file
    model.save(out_path)
    print(f"✅ Saved new model: {out_path}")
