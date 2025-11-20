import pickle
import os

# Path to the folder where you saved your scaler and label encoder
DATA_PATH = "../Preprocessing/CGAN/CGAN_preprocessed_data/"

print(f"Loading Label Encoder from {DATA_PATH}...")

try:
    with open(os.path.join(DATA_PATH, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
        
    print("\n--- Attack Label Mapping ---")
    for index, label in enumerate(le.classes_):
        print(f"Label {index} : {label}")
        
except FileNotFoundError:
    print("Error: Could not find 'label_encoder.pkl'. Make sure the path is correct.")