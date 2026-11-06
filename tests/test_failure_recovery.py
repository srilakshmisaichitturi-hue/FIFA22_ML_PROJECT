import os
import pandas as pd
import numpy as np

def validate_data(df: pd.DataFrame):
    """Checks the dataframe and returns a list of detected errors."""
    errors = []
    
    # Check 1: Target column cannot have missing/null values
    if "overall" in df.columns and df["overall"].isnull().any():
        errors.append("Missing values detected in target column 'overall'")
        
    # Check 2: Stat column must not contain invalid non-numeric text
    stat_col = "pace" if "pace" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    if not pd.to_numeric(df[stat_col], errors="coerce").notnull().all():
        errors.append(f"Non-numeric values found in column '{stat_col}'")
        
    return errors

def test_failure_catch_and_fallback():
    corrupted_file = "artifacts/corrupted_simulation_errors.csv"
    assert os.path.exists(corrupted_file), "Corrupted artifact missing! Run run_lab9_failure_handling.py first."

    df_corrupted = pd.read_csv(corrupted_file)

    # 1. Verify that validation catches the injected corruptions
    detected_errors = validate_data(df_corrupted)
    assert len(detected_errors) > 0, "Validation failed to detect synthetic corruptions!"

    # 2. Automated Recovery / Fallback Logic:
    # A. Drop rows where target 'overall' is missing
    recovered_df = df_corrupted.dropna(subset=["overall"]).copy()
    
    # B. Fix corrupted strings in the numeric stat column by converting to NaN and filling with median
    stat_col = "pace" if "pace" in recovered_df.columns else recovered_df.columns[0]
    recovered_df[stat_col] = pd.to_numeric(recovered_df[stat_col], errors="coerce")
    recovered_df[stat_col] = recovered_df[stat_col].fillna(recovered_df[stat_col].median())

    # 3. Verify clean state after recovery
    recovery_errors = validate_data(recovered_df)
    assert len(recovery_errors) == 0, f"Recovery logic failed to clean data: {recovery_errors}"