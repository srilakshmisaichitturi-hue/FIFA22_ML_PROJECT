import os
import sys
import subprocess
import pandas as pd

# Ensure the root project directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.detect_drift import detect_drift

def check_and_retrain():
    # 1. Locate baseline data
    raw_path = "data/fifa22_players.xlsx"
    if not os.path.exists(raw_path):
        raw_path = "data/raw/fifa22_players.xlsx"

    prod_path = "data/raw/new_players.csv"

    # 2. If the simulated production dataset does not exist, create it now
    if not os.path.exists(prod_path):
        print(f"[LAB 12] Generating simulated production dataset: {prod_path}")
        os.makedirs("data/raw", exist_ok=True)
        df = pd.read_excel(raw_path)
        drifted_df = df.copy()

        # Shift feature values to trigger statistical drift
        if "pace" in drifted_df.columns:
            drifted_df["pace"] = drifted_df["pace"] * 0.80
        if "shooting" in drifted_df.columns:
            drifted_df["shooting"] = drifted_df["shooting"] * 1.20

        drifted_df.to_csv(prod_path, index=False)
        print(f"[LAB 12] Created {prod_path} successfully.")

    # 3. Run statistical drift analysis
    drift_detected = detect_drift(raw_path, prod_path)

    # 4. Trigger retraining pipeline if drift was detected
    if drift_detected:
        print("[LAB 12] Drift detected! Triggering pipeline retraining...")
        subprocess.run([sys.executable, "pipelines/run_lab7_orchestration.py"], check=True)
        print("[LAB 12] Pipeline retrained and models refreshed successfully.")
    else:
        print("[LAB 12] Distribution stable. No retraining required.")

if __name__ == "__main__":
    check_and_retrain()