import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_experiment_tracking():
    print("[INFO] Starting Clean MLflow Experiment Tracking...")
    mlflow.set_experiment("FIFA22_Rating_Prediction_Lab")
    
    models_dir = "models"
    test_data_path = os.path.join(models_dir, "test_data.pkl")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError("test_data.pkl not found. Please run: python pipelines/run_lab3_baseline.py first.")
        
    X_test, X_test_scaled, y_test = joblib.load(test_data_path)
    model_names = ["LinearRegression", "Ridge", "RandomForest"]
    
    for name in model_names:
        with mlflow.start_run(run_name=f"{name}_Clean"):
            model_path = os.path.join(models_dir, f"{name}.pkl")
            model = joblib.load(model_path)
            
            # Predict using the appropriate feature representation
            preds = model.predict(X_test_scaled if name in ["LinearRegression", "Ridge"] else X_test)
            
            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))
            
            # 1. Log Parameters (Experiment 4)
            mlflow.log_param("model_type", name)
            mlflow.log_param("algorithm", model.__class__.__name__)
            
            # 2. Log Metrics (Experiment 4)
            mlflow.log_metric("MAE", round(mae, 4))
            mlflow.log_metric("RMSE", round(rmse, 4))
            mlflow.log_metric("R2_Score", round(r2, 4))
            
            # 3. Log Model Artifact
            try:
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
            except Exception as e:
                print(f"[WARNING] MLflow model artifact logging skipped: {e}")
                
            print(f"[SUCCESS] {name}_Clean logged: R2={r2:.4f}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    run_experiment_tracking()
