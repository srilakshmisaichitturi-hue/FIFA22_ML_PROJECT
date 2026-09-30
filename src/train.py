import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

def train_models(data_path="data/processed/clean_players.csv", models_dir="models"):
    print("[INFO] Running Model Training...")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=["overall"])
    y = df["overall"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    for name, model in models.items():
        if name in ["LinearRegression", "Ridge"]:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))
        
    joblib.dump((X_test, X_test_scaled, y_test), os.path.join(models_dir, "test_data.pkl"))
    print("[SUCCESS] All baseline models trained and saved to /models.")

if __name__ == "__main__":
    train_models()
