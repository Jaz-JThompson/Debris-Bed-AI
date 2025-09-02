import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
import os

# Safety check: run only in TF 2.15
if tf.__version__ != "2.15.0":
    raise RuntimeError(f"Script must be run in TensorFlow 2.15, found {tf.__version__}")

# Paths to old models
old_model_paths = [f"model_{i+1}.keras" for i in range(5)]
new_model_paths = [f"model_new_{i+1}.keras" for i in range(5)]

for old_path, new_path in zip(old_model_paths, new_model_paths):
    print(f"Processing {old_path} → {new_path}")

    # Load original model without compiling
    old_model = load_model(old_path, compile=False)

    # Save in TF 2.20 compatible format by re-saving
    # This avoids manually reconstructing layers
    old_model.save(new_path)
    print(f"Saved safely as {new_path}")
