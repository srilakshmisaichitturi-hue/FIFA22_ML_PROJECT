import os
import pandas as pd

def run_lab1():
    print('='*60)
    print('  LAB 1: DATA INGESTION & OUTFIELD FEATURE EXTRACTION')
    print('='*60)
    
    raw_path = os.path.join('data', 'raw', 'fifa22_players.xlsx')
    if not os.path.exists(raw_path):
        raw_path = os.path.join('data', 'raw', 'clean_players.csv')
        df = pd.read_csv(raw_path)
    else:
        df = pd.read_excel(raw_path)
        
    print(f'[+] Raw Data Loaded Successfully. Shape: {df.shape}')
    
    features = [
        'potential', 'value_eur', 'wage_eur', 'age', 'height_cm', 'weight_kg',
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'attacking_crossing', 'attacking_finishing', 'attacking_short_passing',
        'skill_dribbling', 'skill_curve', 'movement_reactions', 'power_shot_power',
        'mentality_vision'
    ]
    target = 'overall'
    
    df_clean = df.dropna(subset=features + [target]).copy()
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    out_csv = os.path.join('data', 'processed', 'clean_players.csv')
    df_clean.to_csv(out_csv, index=False)
    print(f'[+] Cleaned dataset saved to: {out_csv}')
    print(f'[+] Cleaned Shape: {df_clean.shape} (Removed outfield nulls)')

if __name__ == '__main__':
    run_lab1()
