import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_lab4():
    print('============================================================')
    print('  LAB 4: EXPERIMENT TRACKING WITH MLFLOW')
    print('============================================================')
    
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('FIFA22_Rating_Prediction_Lab')
    
    csv_path = os.path.join('data', 'processed', 'clean_players.csv')
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=['overall']).select_dtypes(include=[np.number])
    y = df['overall']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_path = os.path.join('models', 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test
    
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and f not in ['scaler.pkl', 'test_data.pkl']]
    
    for filename in model_files:
        model_name = os.path.splitext(filename)[0]
        model_path = os.path.join('models', filename)
        
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f'[-] Skipping {filename}: {e}')
            continue
            
        # Linear models use scaled inputs; tree models use raw unscaled inputs
        if 'RandomForest' in model_name:
            preds = model.predict(X_test)
        else:
            preds = model.predict(X_test_scaled)
            
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        
        with mlflow.start_run(run_name=f'{model_name}_Tracking'):
            mlflow.log_param('model_name', model_name)
            mlflow.log_metric('MAE', mae)
            mlflow.log_metric('RMSE', rmse)
            mlflow.log_metric('R2_Score', r2)
            
            try:
                mlflow.sklearn.log_model(
                    model, 
                    name='model',
                    skops_trusted_types=['sklearn.tree._tree.Tree']
                )
            except Exception:
                try:
                    mlflow.sklearn.log_model(model, name='model')
                except Exception:
                    pass
                    
            print(f'[+] Logged {model_name:<18} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}')

if __name__ == '__main__':
    run_lab4()
