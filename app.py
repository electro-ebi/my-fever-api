from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("fever_model.pkl")

# Map prediction to extra information
info_map = {
    "Viral Fever": ("Low", "Maybe", "Paracetamol, Rest"),
    "Bacterial Infection": ("High", "Yes", "Antibiotics"),
    "Typhoid": ("Moderate", "Yes", "Antibiotics, Paracetamol"),
    "Dengue": ("High", "Yes", "Fluid intake, Paracetamol (NO NSAIDs)"),
    "Normal": ("Low", "No", "None")
}


@app.route("/")
def home():
    return "🔥 Fever Detection API is Live! Use POST /predict"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()

        # Convert to DataFrame for model
        df = pd.DataFrame([data])

        # Get prediction
        prediction = model.predict(df)[0]

        # Get mapped info
        severity, consult, medicine = info_map.get(prediction, ("Unknown", "Unknown", "Unknown"))

        # Return result
        return jsonify({
            "prediction": prediction,
            "severity": severity,
            "consult_doctor": consult,
            "suggested_medicine": medicine,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
