import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import load_and_preprocess

if __name__ == '__main__':
    print('[RUNNING LAB 5: DATA VALIDATION & PREPROCESSING PIPELINE]')
    load_and_preprocess()
