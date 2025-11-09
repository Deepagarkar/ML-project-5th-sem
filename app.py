# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

# expected feature names - match your dataset order
FEATURE_NAMES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # read values from form in same order as FEATURE_NAMES
        values = []
        for f in FEATURE_NAMES:
            v = request.form.get(f)
            if v is None or v.strip() == "":
                return render_template("result.html", error=f"Missing value for {f}")
            values.append(float(v))
        x = np.array([values])
        pred = model.predict(x)[0]
        proba = float(model.predict_proba(x)[0][1])  # probability of class 1 (diabetic)
        result_text = "Positive for Diabetes (High risk)" if pred == 1 else "Negative for Diabetes (Low risk)"
        return render_template("result.html", result=result_text, proba=round(proba, 3))
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
