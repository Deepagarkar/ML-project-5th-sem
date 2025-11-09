from flask import Flask, render_template, request
import joblib, os, numpy as np

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# If model missing, try to train automatically (needs diabetes.csv present)
if not os.path.exists(MODEL_PATH):
    try:
        # lazy import to avoid import errors if sklearn missing at build time
        from train_model import train_and_save
        train_and_save(MODEL_PATH)
    except Exception as e:
        # raise error so Render logs show why
        raise RuntimeError("Model missing and auto-train failed: " + str(e))

model = joblib.load(MODEL_PATH)
FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        vals = [float(request.form.get(f)) for f in FEATURES]
        pred = int(model.predict([vals])[0])
        prob = float(model.predict_proba([vals])[0][1])
        result = "High Risk (Positive)" if pred==1 else "Low Risk (Negative)"
        return render_template("result.html", result=result, prob=round(prob,3))
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
