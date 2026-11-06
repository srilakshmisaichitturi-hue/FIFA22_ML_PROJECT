# Experiment 13: End-to-End MLOps System Mini Project

## 1. System Overview
The FIFA 22 Player Overall Rating Predictor is an end-to-end MLOps pipeline covering:
- **Data Engineering**: Handling tabular player data, imputing null values, and scaling features.
- **Experiment Tracking**: Training regression baselines (Random Forest, Ridge, Linear Regression) logged via MLflow.
- **Automation & Quality Gates**: Ensuring RMSE < 3.5 and $R^2$ > 0.85 prior to artifact serialization.
- **CI/CD Integration**: Automated GitHub Actions workflows for continuous integration and fault-injection testing.
- **Model Serving**: High-throughput FastAPI REST API serving player predictions with Pydantic validation.
- **Packaging**: Containerized deployment via Docker and Docker Compose.
- **Monitoring**: Statistical data drift detection utilizing the two-sample Kolmogorov-Smirnov test with automated retraining triggers.

## 2. Component Pipeline Map
1. `pipelines/run_lab1_dataprep.py` -> Dataset ingestion & preprocessing
2. `pipelines/run_lab4_tracking.py` -> MLflow parameter & metric tracking
3. `pipelines/run_lab7_orchestration.py` -> Model evaluation & serialization
4. `tests/` -> Automated test coverage (CI & Failure handling)
5. `app/main.py` -> FastAPI production endpoints (`/health`, `/predict`, `/docs`)
6. `monitoring/detect_drift.py` -> Two-sample KS test drift analysis
7. `pipelines/run_lab13_e2e_system.py` -> Master orchestrator connecting all components

## 3. Execution Verification
The complete system was validated by executing `pipelines/run_lab13_e2e_system.py`. 
Audit outputs and status records are archived in `reports/e2e_system_report.json`.