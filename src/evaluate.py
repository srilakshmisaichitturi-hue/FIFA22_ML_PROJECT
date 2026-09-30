import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(models_dir="models"):
    print("[INFO] Evaluating Models...")
    test_path = os.path.join(models_dir, "test_data.pkl")
    X_test, X_test_scaled, y_test = joblib.load(test_path)
    
    model_names = ["LinearRegression", "Ridge", "RandomForest"]
    results = {}
    
    for name in model_names:
        model = joblib.load(os.path.join(models_dir, f"{name}.pkl"))
        preds = model.predict(X_test_scaled if name in ["LinearRegression", "Ridge"] else X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"Model: {name:<18} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
    return results

if __name__ == "__main__":
    evaluate_models()
