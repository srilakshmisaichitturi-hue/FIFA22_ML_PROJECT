import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import train_models

if __name__ == '__main__':
    print('[RUNNING LAB 6: MODEL TRAINING & SERIALIZATION PIPELINE]')
    train_models()
