import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

def run_lab6():
    print('='*60)
    print('  LAB 6: MODEL REGISTRY & SERIALIZATION ARTIFACTS')
    print('='*60)
    
    csv_path = os.path.join('data', 'processed', 'clean_players.csv')
    df = pd.read_csv(csv_path)
    
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
    
    os.makedirs('models', exist_ok=True)
    with open(os.path.join('models', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join('models', 'test_data.pkl'), 'wb') as f:
        pickle.dump({'X_test': X_test_scaled, 'y_test': y_test}, f)
        
    models = {
        'LinearRegression.pkl': LinearRegression(),
        'Ridge.pkl': Ridge(alpha=1.0),
        'RandomForest.pkl': RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42)
    }
    
    for filename, model in models.items():
        model.fit(X_train_scaled, y_train)
        path = os.path.join('models', filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f'[+] Serialized artifact saved: {path}')

if __name__ == '__main__':
    run_lab6()
