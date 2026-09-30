import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_and_preprocess
from src.train import train_models
from src.evaluate import evaluate_models

def run_pipeline():
    print("=" * 60)
    print("EXECUTING EXPERIMENT 7: MODULAR ML PIPELINE")
    print("=" * 60)
    
    clean_path = load_and_preprocess()
    train_models(data_path=clean_path)
    evaluate_models()
    
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
