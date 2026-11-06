import os
import joblib
import pandas as pd
import numpy as np

def test_artifacts_exist():
    """Verify that required model artifacts from earlier labs exist."""
    assert os.path.exists("models/final_fifa22_model.pkl"), "Trained model pkl is missing!"
    assert os.path.exists("models/scaler.pkl"), "Scaler pkl is missing!"

def test_model_prediction_shape():
    """Check that the model can load and predict on valid synthetic data."""
    model = joblib.load("models/final_fifa22_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Assuming numerical stat input of size matching your scaler
    n_features = scaler.n_features_in_
    sample_input = np.ones((1, n_features)) * 75.0  # mock rating values

    scaled_input = scaler.transform(sample_input)
    prediction = model.predict(scaled_input)

    assert len(prediction) == 1
    assert 40 <= prediction[0] <= 99, f"Unexpected predicted rating: {prediction[0]}"