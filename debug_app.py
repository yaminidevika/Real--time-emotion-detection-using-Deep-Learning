import cv2
import os
import sys
from keras.models import load_model

log_file = "debug_log.txt"

def log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    print(msg)

# Clear log
with open(log_file, "w") as f:
    f.write("Starting Debug...\n")

# Check CWD
cwd = os.getcwd()
log(f"Current Working Directory: {cwd}")

# 1. Check Face Cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
log(f"Attempting to load cascade from: {cascade_path}")

if not os.path.exists(cascade_path):
    log("ERROR: Cascade file does not exist at path.")
else:
    log("Cascade file exists.")

try:
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        log("ERROR: Failed to load Face Cascade (empty).")
    else:
        log("SUCCESS: Face Cascade loaded.")
except Exception as e:
    log(f"ERROR: Exception loading cascade: {e}")

# 2. Check Keras Model
model_path = 'emotion_detector_models/model_v6_23.hdf5'
log(f"Attempting to load model from: {model_path}")

if not os.path.exists(model_path):
    log("ERROR: Model file does not exist at path.")
else:
    log("Model file exists.")

try:
    # Try loading with compile=False first as it's safer for deployment if we just predict
    model = load_model(model_path, compile=False)
    log("SUCCESS: Model loaded (compile=False).")
except Exception as e:
    log(f"ERROR: Exception loading model (compile=False): {e}")
    try:
        model = load_model(model_path)
        log("SUCCESS: Model loaded (compile=True).")
    except Exception as e2:
        log(f"ERROR: Exception loading model (compile=True): {e2}")

log("Debug complete.")
