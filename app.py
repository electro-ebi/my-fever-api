from flask import Flask, render_template, request
import pandas as pd
import joblib

model = joblib.load("fever_model.pkl")

info_map = {
    "Bacterial Infection": ["High", "Yes", "Antibiotics"],
    "Viral Fever": ["Low", "Maybe", "Paracetamol, Rest"],
    "Typhoid": ["Moderate", "Yes", "Antibiotics, Paracetamol"],
    "Normal": ["Low", "No", "None"],
    "Dengue": ["High", "Yes", "Fluid Replacement, Paracetamol"]
}


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        features = [float(request.form.get(field)) for field in [
            "Hemoglobin", "WBC", "Platelets", "Neutrophils", "Lymphocytes",
            "Monocytes", "Eosinophils", "Basophils", "MCHC", "RDW", "PCV"
        ]]
        df = pd.DataFrame([features], columns=[
            "Hemoglobin", "WBC", "Platelets", "Neutrophils", "Lymphocytes",
            "Monocytes", "Eosinophils", "Basophils", "MCHC", "RDW", "PCV"
        ])
        prediction = model.predict(df)[0]
        severity, consult, medicine = info_map[prediction]
        result = {
            "Fever Type": prediction,
            "Severity": severity,
            "Consult": consult,
            "Medicine": medicine
        }
    return render_template("form.html", result=result)

# ✅ Make sure this part is included
if __name__ == "__main__":
    app.run(debug=True)
