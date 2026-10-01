import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_lab4():
    print('='*60)
    print('  LAB 4: EXPERIMENT TRACKING WITH MLFLOW')
    print('='*60)
    
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('FIFA22_Rating_Prediction_Lab')
    
    with open(os.path.join('models', 'test_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    X_test, y_test = data['X_test'], data['y_test']
    
    model_names = ['LinearRegression', 'Ridge', 'RandomForest']
    for name in model_names:
        model_path = os.path.join('models', f'{name}.pkl')
        if not os.path.exists(model_path):
            continue
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        with mlflow.start_run(run_name=f'{name}_Clean'):
            mlflow.log_param('model_type', name)
            mlflow.log_metric('MAE', mae)
            mlflow.log_metric('RMSE', rmse)
            mlflow.log_metric('R2_Score', r2)
            mlflow.sklearn.log_model(model, 'model')
            print(f'[+] Logged {name} to MLflow | R2: {r2:.4f}')

if __name__ == '__main__':
    run_lab4()
