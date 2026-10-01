import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_lab2():
    print('='*60)
    print('  LAB 2: BASELINE REGRESSION MODEL TRAINING & EVALUATION')
    print('='*60)
    
    data_path = os.path.join('data', 'processed', 'clean_players.csv')
    df = pd.read_csv(data_path)
    
    features = [
        'potential', 'value_eur', 'wage_eur', 'age', 'height_cm', 'weight_kg',
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'attacking_crossing', 'attacking_finishing', 'attacking_short_passing',
        'skill_dribbling', 'skill_curve', 'movement_reactions', 'power_shot_power',
        'mentality_vision'
    ]
    target = 'overall'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42)
    }
    
    print(f'Evaluated on Test Split ({X_test.shape[0]} players):')
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f'  -> {name:<18} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}')

if __name__ == '__main__':
    run_lab2()
