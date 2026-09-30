import os
import json
import pandas as pd

FEATURES = [
    "age", "height_cm", "weight_kg", "pace", "shooting", "passing", 
    "dribbling", "defending", "physic", "attacking_crossing", 
    "attacking_finishing", "skill_dribbling", "skill_ball_control", 
    "power_shot_power", "power_stamina", "power_strength", 
    "mentality_aggression", "mentality_interceptions", 
    "mentality_positioning", "mentality_vision"
]
TARGET = "overall"

def load_and_preprocess(input_path="data/raw/fifa22_players.xlsx", output_dir="data/processed"):
    print("[INFO] Running Data Preprocessing & Validation...")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
        
    df = pd.read_excel(input_path)
    
    required_cols = FEATURES + [TARGET]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema mismatch: missing columns: {missing_cols}")
        
    clean_df = df[required_cols].dropna()
    
    os.makedirs(output_dir, exist_ok=True)
    clean_csv_path = os.path.join(output_dir, "clean_players.csv")
    clean_df.to_csv(clean_csv_path, index=False)
    
    metadata = {
        "dataset_name": "FIFA 22 Players",
        "raw_record_count": int(len(df)),
        "processed_record_count": int(len(clean_df)),
        "features": FEATURES,
        "target": TARGET
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[SUCCESS] Clean dataset saved ({len(clean_df)} records) with metadata.")
    return clean_csv_path

if __name__ == "__main__":
    load_and_preprocess()
