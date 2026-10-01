import subprocess
import sys

def run_lab7():
    print('='*60)
    print('  LAB 7: MASTER END-TO-END PIPELINE ORCHESTRATION')
    print('='*60)
    
    stages = [
        ('Stage 1: Data Ingestion (Lab 1)', 'pipelines/run_lab1_dataprep.py'),
        ('Stage 2: Data Validation (Lab 5)', 'pipelines/run_lab5_data_validation.py'),
        ('Stage 3: Model Training & Serialization (Lab 6)', 'pipelines/run_lab6_model_serialization.py'),
        ('Stage 4: Baseline Evaluation (Lab 2)', 'pipelines/run_lab2_baselines.py'),
        ('Stage 5: MLflow Tracking & Registry (Lab 4)', 'pipelines/run_lab4_tracking.py')
    ]
    
    for desc, script in stages:
        print(f'\n>>> Executing {desc}...')
        res = subprocess.run([sys.executable, script], check=True)
        
    print('\n' + '='*60)
    print('  [COMPLETED] FULL MLOPS PIPELINE EXECUTED SUCCESSFULLY!')
    print('='*60)

if __name__ == '__main__':
    run_lab7()
