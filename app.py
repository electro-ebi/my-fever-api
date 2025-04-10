from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("fever_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features in order (ensure same order as model was trained)
        features = [
            data['Hemoglobin'], data['WBC'], data['Platelets'],
            data['Neutrophils'], data['Lymphocytes'], data['Monocytes'],
            data['Eosinophils'], data['Basophils'], data['MCHC'],
            data['RDW'], data['PCV']
        ]

        prediction = model.predict([features])[0]
        return jsonify({
            "prediction": prediction,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
