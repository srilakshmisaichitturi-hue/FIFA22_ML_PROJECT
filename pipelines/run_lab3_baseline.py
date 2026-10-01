import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import load_and_preprocess
from src.train import train_models
from src.evaluate import evaluate_models

def run_lab3():
    print('='*60)
    print('  LAB 3: BASELINE MODULAR PIPELINE EXECUTION')
    print('='*60)
    load_and_preprocess()
    train_models()
    evaluate_models()

if __name__ == '__main__':
    run_lab3()
