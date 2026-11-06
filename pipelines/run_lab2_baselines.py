import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import train_models
from src.evaluate import evaluate_models

if __name__ == '__main__':
    print('============================================================')
    print('  LAB 2: BASELINE REGRESSION MODEL EVALUATION')
    print('============================================================')
    train_models()
    evaluate_models()
