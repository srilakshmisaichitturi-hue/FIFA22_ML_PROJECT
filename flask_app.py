from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the final selected machine learning model
model = joblib.load("models/final_fifa22_model.pkl")

# Features used during model training
features = [
    "age",
    "height_cm",
    "weight_kg",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "attacking_crossing",
    "attacking_finishing",
    "skill_dribbling",
    "skill_ball_control",
    "power_shot_power",
    "power_stamina",
    "power_strength",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision"
]


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Check whether all required features are provided
        missing_features = [
            feature for feature in features
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing features: " + ", ".join(missing_features)
            }), 400

        # Create input DataFrame
        input_data = pd.DataFrame(
            [[data[feature] for feature in features]],
            columns=features
        )

        # Make prediction
        prediction = model.predict(input_data)

        # Return prediction
        return jsonify({
            "predicted_overall": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Start Flask application
if __name__ == "__main__":
    app.run(debug=True)