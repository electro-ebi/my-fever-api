from flask import Flask, request, jsonify, render_template
import requests
import datetime
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("fever_model.pkl")

# 1. --- TEMP CHECKUP ---
@app.route("/check_temp", methods=["GET"])
def check_temp():
    THINGSPEAK_URL = "https://api.thingspeak.com/channels/2914501/feeds.json?api_key=2HLDF9ZWYNV4YWUG&results=10"

    response = requests.get(THINGSPEAK_URL)
    data = response.json()
    feeds = data.get("feeds", [])

    if not feeds:
        return jsonify({"status": "error", "message": "No temperature data found"})

    temps = [float(f["field1"]) for f in feeds if f["field1"]]
    if not temps:
        return jsonify({"status": "error", "message": "No object temps found"})

    avg_temp = sum(temps) / len(temps)
    fever_detected = avg_temp >= 37.5

    return jsonify({
        "avg_temp": round(avg_temp, 2),
        "fever": fever_detected,
        "status": "success"
    })

# 2. --- FEVER PREDICTION ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = pd.DataFrame([data])

        prediction = model.predict(features)[0]

        info_map = {
            "Bacterial Infection": ["High", "Yes", "Antibiotics"],
            "Viral Fever": ["Low", "Maybe", "Paracetamol, Rest"],
            "Typhoid": ["Moderate", "Yes", "Antibiotics, Paracetamol"],
            "Normal": ["Low", "No", "None"],
            "Dengue": ["High", "Yes", "Paracetamol, Fluids"]
        }

        severity, consult, medicine = info_map.get(prediction, ("Unknown", "Maybe", "Consult Doctor"))

        return jsonify({
            "prediction": prediction,
            "severity": severity,
            "consult_doctor": consult,
            "suggested_medicine": medicine,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"})


# Local test view (optional)
@app.route("/")
def home():
    return render_template("index.html")  # Make sure index.html is placed inside a /templates folder

if __name__ == "__main__":
    app.run(debug=True)
