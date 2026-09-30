import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_experiment_tracking():
    print("[INFO] Starting MLflow Experiment Tracking...")
    mlflow.set_experiment("FIFA22_Rating_Prediction_Lab")
    
    models_dir = "models"
    X_test, X_test_scaled, y_test = joblib.load(os.path.join(models_dir, "test_data.pkl"))
    model_names = ["LinearRegression", "Ridge", "RandomForest"]
    
    for name in model_names:
        with mlflow.start_run(run_name=name):
            model = joblib.load(os.path.join(models_dir, f"{name}.pkl"))
            preds = model.predict(X_test_scaled if name in ["LinearRegression", "Ridge"] else X_test)
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            mlflow.log_param("model_type", name)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2_Score", r2)
            mlflow.sklearn.log_model(model, artifact_path=name)
            
            print(f"[MLflow] Logged run for {name}: R2={r2:.4f}")

if __name__ == "__main__":
    run_experiment_tracking()
