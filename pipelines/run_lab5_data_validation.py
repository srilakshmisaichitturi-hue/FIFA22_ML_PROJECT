import os
import json
import pandas as pd

def run_lab5():
    print('='*60)
    print('  LAB 5: DATA SCHEMA VALIDATION & INTEGRITY CHECKS')
    print('='*60)
    
    csv_path = os.path.join('data', 'processed', 'clean_players.csv')
    df = pd.read_csv(csv_path)
    
    required_cols = [
        'potential', 'value_eur', 'wage_eur', 'age', 'height_cm', 'weight_kg',
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'attacking_crossing', 'attacking_finishing', 'attacking_short_passing',
        'skill_dribbling', 'skill_curve', 'movement_reactions', 'power_shot_power',
        'mentality_vision', 'overall'
    ]
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    assert len(missing_cols) == 0, f'Missing required columns: {missing_cols}'
    assert df['overall'].between(40, 100).all(), 'Target overall rating out of valid FIFA bounds [40, 100]!'
    assert df.isnull().sum().sum() == 0, 'Processed data contains null values!'
    
    metadata = {
        'total_records': int(len(df)),
        'total_features': len(required_cols) - 1,
        'target_variable': 'overall',
        'status': 'VALIDATED_PASSED',
        'columns': {col: str(df[col].dtype) for col in required_cols}
    }
    
    meta_path = os.path.join('data', 'processed', 'dataset_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f'[+] Data validation PASSED successfully.')
    print(f'[+] Generated schema verification at: {meta_path}')

if __name__ == '__main__':
    run_lab5()
