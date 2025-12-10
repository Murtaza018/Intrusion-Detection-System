# check_label_encoder.py
# Diagnostic tool to inspect the contents of your Label Encoder
# FIXED: Resolved UnboundLocalError by removing variable scope ambiguity.

import joblib
import os
import sys

def check_encoder():
    print(f"--- 🔍 Inspecting Label Encoder ---")
    
    # 1. Define the filename we are looking for
    filename = "label_encoder.pkl"
    target_path = None

    # 2. Check multiple locations
    possible_locations = [
        os.getcwd(),                                      # Current folder
        os.path.join(os.getcwd(), "ids_pipeline"),        # Subfolder
        os.path.dirname(os.path.abspath(__file__))        # Script's own folder
    ]

    for loc in possible_locations:
        check_path = os.path.join(loc, filename)
        if os.path.exists(check_path):
            target_path = check_path
            break
    
    if not target_path:
        print(f"[!] Error: Could not find '{filename}' in any expected directory.")
        print(f"    Searched in: {possible_locations}")
        return

    print(f"[*] Found file at: {target_path}")

    try:
        # 3. Load the object
        encoder = joblib.load(target_path)
        print(f"[+] Successfully loaded object.")
        print(f"    Type: {type(encoder)}")

        # 4. Check contents
        if hasattr(encoder, 'classes_'):
            classes = encoder.classes_
            print(f"\n[+] Valid LabelEncoder detected!")
            print(f"    Total Classes: {len(classes)}")
            print("-" * 30)
            
            # Print class mapping
            for i, class_name in enumerate(classes):
                print(f"    ID {i}: {class_name}")
            
            print("-" * 30)
            print("✅ VERDICT: This encoder is SAFE to use.")
            
        elif isinstance(encoder, dict):
            print(f"\n[+] Dictionary-based mapping detected.")
            print(f"    Keys: {list(encoder.keys())}")
            print("⚠️ VERDICT: Usable, but requires custom handling.")
            
        else:
            print(f"\n[!] Unknown object structure: {dir(encoder)}")
            print("❌ VERDICT: Do NOT use.")

    except Exception as e:
        print(f"\n[!] Critical Error loading file: {e}")

if __name__ == "__main__":
    check_encoder()