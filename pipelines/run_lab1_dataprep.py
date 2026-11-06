import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import load_and_preprocess

if __name__ == '__main__':
    print('============================================================')
    print('  LAB 1: DATA PREPARATION & INGESTION')
    print('============================================================')
    load_and_preprocess()
