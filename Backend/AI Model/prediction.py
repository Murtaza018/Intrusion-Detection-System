# ids_pipeline.py
# Description: This script demonstrates the final, complete hybrid pipeline. It uses our
# most powerful model (the Spatio-Temporal CNN+LSTM) as the frontline defense and
# our best unsupervised model (the Deep Autoencoder) as the zero-day hunter.

import numpy as np
from tensorflow import keras

print("--- Initializing the Final Hybrid IDS Pipeline ---")

# --- Step 1: Load Our Two Final, Trained Models ---
try:
    print("[*] Loading the Spatio-Temporal Classifier (Frontline Defense)...")
    # ** THE UPGRADE **: We are now using our best and most powerful model.
    classifier_model = keras.models.load_model("./CNN+LSTM/cicids_spatiotemporal_model.keras")
    print("[+] Spatio-Temporal model loaded successfully.")

    print("[*] Loading the Autoencoder (Zero-Day Hunter)...")
    autoencoder_model = keras.models.load_model("./Autoencoder/cicids_autoencoder.keras")
    print("[+] Autoencoder loaded successfully.")

except Exception as e:
    print(f"\n[!] Error loading one or more models: {e}")
    print("[!] Ensure 'cicids_spatiotemporal_model.keras' and 'cicids_autoencoder.keras' are present.")
    exit()

# --- Step 2: Define the Anomaly Threshold ---
AUTOENCODER_THRESHOLD = 0.0001 # From our autoencoder training.

# --- Step 3: Create the Prediction Pipeline Logic ---
def predict_traffic(data_sample):
    """
    Analyzes a single sample of network traffic using our final hybrid model pipeline.
    """
    # The models require different input shapes, so we create a version for each.
    # Reshape for the supervised model, which expects a 3D input: (1, 78, 1)
    if data_sample.ndim == 1:
        supervised_sample = np.reshape(data_sample, (1, -1, 1))
    else: # If already batched
        supervised_sample = np.expand_dims(data_sample, -1)

    # Reshape for the autoencoder, which expects a 2D input: (1, 78)
    autoencoder_sample = np.reshape(data_sample, (1, -1))


    # --- Pipeline Stage 1: Spatio-Temporal Classifier ---
    # This is our fast and accurate frontline defense.
    classifier_prediction_proba = classifier_model.predict(supervised_sample, verbose=0)[0][0]

    if classifier_prediction_proba > 0.5:
        prediction = "Malicious (Known Attack)"
        explanation = (
            f"The advanced CNN+LSTM classifier is {classifier_prediction_proba:.2%} confident "
            "that this traffic matches a known attack pattern."
        )
        return prediction, explanation

    # --- Pipeline Stage 2: Unsupervised Anomaly Hunter ---
    # Only if the first model clears the traffic do we check for novel anomalies.
    reconstruction = autoencoder_model.predict(autoencoder_sample, verbose=0)
    reconstruction_error = np.mean(np.abs(autoencoder_sample - reconstruction))

    if reconstruction_error > AUTOENCODER_THRESHOLD:
        prediction = "Anomalous (Potential Zero-Day)"
        explanation = (
            "The traffic was cleared by the main classifier, but the autoencoder detected "
            f"a high reconstruction error of {reconstruction_error:.6f} (threshold: {AUTOENCODER_THRESHOLD:.6f}). "
            "This indicates a deviation from normal patterns and should be investigated."
        )
        return prediction, explanation

    # --- Final Verdict: Normal ---
    # If it passes both checks, we can be very confident it's safe.
    prediction = "Benign (Normal)"
    explanation = (
        "The traffic was cleared by the advanced classifier and showed a low reconstruction "
        f"error of {reconstruction_error:.6f}, indicating it conforms to normal patterns."
    )
    return prediction, explanation

# --- Step 4: Example Usage ---
if __name__ == "__main__":
    print("\n--- Running a test prediction on a sample from the dataset ---")
    
    # Per your instruction, using the confirmed correct data path.
    DATA_PATH = "./Preprocessing/CIC-IDS-2017/CIC-IDS-2017-Processed/"
    try:
        X_test = np.load(DATA_PATH + 'X_test.npy')
        y_test = np.load(DATA_PATH + 'y_test.npy')
    except FileNotFoundError:
        print(f"\n[!] Error: Could not find test data at '{DATA_PATH}'.")
        print("[!] Please ensure the preprocessed files are in the correct location.")
        exit()

    
    # Example 1: A known attack sample
    print("\n--- Test Case 1: Analyzing a known attack... ---")
    attack_sample = X_test[y_test == 1][0] 
    pred, expl = predict_traffic(attack_sample)
    print(f"  Prediction: {pred}")
    print(f"  Explanation: {expl}")

    # Example 2: A known normal sample
    print("\n--- Test Case 2: Analyzing a known normal sample... ---")
    normal_sample = X_test[y_test == 0][0] 
    pred, expl = predict_traffic(normal_sample)
    print(f"  Prediction: {pred}")
    print(f"  Explanation: {expl}")

