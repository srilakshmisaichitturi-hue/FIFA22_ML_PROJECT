import os, json, pandas as pd

def run_lab5():
    print('============================================================')
    print('  LAB 5: DATA SCHEMA VALIDATION & INTEGRITY CHECKS')
    print('============================================================')
    csv_path = os.path.join('data', 'processed', 'clean_players.csv')
    df = pd.read_csv(csv_path)
    assert 'overall' in df.columns, 'Target column overall is missing!'
    assert df['overall'].between(40, 100).all(), 'Target overall out of bounds!'
    assert df.isnull().sum().sum() == 0, 'Processed data contains null values!'
    
    metadata = {
        'total_records': int(len(df)),
        'total_features': int(df.shape[1] - 1),
        'target_variable': 'overall',
        'status': 'VALIDATED_PASSED',
        'columns': {col: str(df[col].dtype) for col in df.columns}
    }
    meta_path = os.path.join('data', 'processed', 'dataset_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f'[+] Data validation PASSED successfully ({len(df)} records).')
    print(f'[+] Metadata schema saved to: {meta_path}')

if __name__ == '__main__':
    run_lab5()
