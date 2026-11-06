import os
import pandas as pd
import numpy as np

def simulate_data_corruption():
    print("[LAB 9] Generating corrupted FIFA dataset simulation...")

    # Look for raw dataset
    raw_path = "data/fifa22_players.xlsx"
    if not os.path.exists(raw_path):
        raw_path = "data/raw/fifa22_players.xlsx"

    df = pd.read_excel(raw_path)
    corrupted_df = df.copy()

    # 1. Inject missing target values
    if "overall" in corrupted_df.columns:
        corrupted_df.loc[0:15, "overall"] = np.nan

    # 2. Inject corrupted non-numeric string into a stat column
    stat_col = "pace" if "pace" in corrupted_df.columns else corrupted_df.select_dtypes(include=[np.number]).columns[0]
    
    # Cast column to object dtype first so pandas allows string insertion
    corrupted_df[stat_col] = corrupted_df[stat_col].astype(object)
    corrupted_df.loc[20:25, stat_col] = "CORRUPTED_VALUE"

    os.makedirs("artifacts", exist_ok=True)
    out_file = "artifacts/corrupted_simulation_errors.csv"
    corrupted_df.to_csv(out_file, index=False)
    print(f"[LAB 9] Corrupted data saved to {out_file}")

if __name__ == "__main__":
    simulate_data_corruption()