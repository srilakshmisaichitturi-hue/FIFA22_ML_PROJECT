import os
import json
import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(baseline_file: str, current_file: str, alpha: float = 0.05):
    print("[LAB 12] Running statistical drift analysis...")
    df_base = pd.read_excel(baseline_file) if baseline_file.endswith(".xlsx") else pd.read_csv(baseline_file)
    df_curr = pd.read_excel(current_file) if current_file.endswith(".xlsx") else pd.read_csv(current_file)

    numerical_cols = [
        col for col in ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
        if col in df_base.columns and col in df_curr.columns
    ]

    metrics = {}
    drift_count = 0

    for col in numerical_cols:
        stat, p_val = ks_2samp(df_base[col].dropna(), df_curr[col].dropna())
        is_drifted = bool(p_val < alpha)
        if is_drifted:
            drift_count += 1

        metrics[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 5),
            "drift_detected": is_drifted
        }

    overall_drift = drift_count >= 2

    os.makedirs("reports", exist_ok=True)
    report = {
        "overall_drift_detected": overall_drift,
        "features_drifted_count": drift_count,
        "feature_metrics": metrics
    }

    with open("reports/drift_metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"[LAB 12] Drift analysis complete. Overall drift detected: {overall_drift}")
    return overall_drift

if __name__ == "__main__":
    raw_path = "data/fifa22_players.xlsx"
    if not os.path.exists(raw_path):
        raw_path = "data/raw/fifa22_players.xlsx"

    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_excel(raw_path)
    drifted_df = df.copy()
    if "pace" in drifted_df.columns:
        drifted_df["pace"] = drifted_df["pace"] * 0.80
    if "shooting" in drifted_df.columns:
        drifted_df["shooting"] = drifted_df["shooting"] * 1.20
    drifted_df.to_csv("data/raw/new_players.csv", index=False)

    detect_drift(raw_path, "data/raw/new_players.csv")