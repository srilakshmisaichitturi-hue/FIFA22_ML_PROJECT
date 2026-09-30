import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipelines.run_lab3_baseline import run_pipeline

if __name__ == '__main__':
    print('[RUNNING LAB 7: END-TO-END ORCHESTRATED ML PIPELINE]')
    run_pipeline()
