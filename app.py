from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# -------------------- TRAIN MODEL --------------------
print("⚙️ Training model (no model.pkl needed)...")

# Load dataset (safe encoding)
data = pd.read_csv("diabetes.csv", encoding='latin1')

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

print("✅ Model trained successfully!")

# -------------------- WEB UI --------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Predictor</title>
    <style>
        body { font-family: Arial; background: linear-gradient(to right, #d0f0f0, #f4e1ff);
               display: flex; flex-direction: column; align-items: center; padding: 30px; }
        h1 { color: #333; }
        form { background: white; padding: 25px; border-radius: 12px; width: 320px; box-shadow: 0 0 15px rgba(0,0,0,0.2); }
        input { width: 90%; padding: 8px; margin: 5px 0; border: 1px solid #ccc; border-radius: 8px; }
        button { background: #6200ea; color: white; border: none; padding: 10px; width: 100%; border-radius: 8px; cursor: pointer; }
        button:hover { background: #3700b3; }
        h2 { color: #111; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🩺 Diabetes Disease Prediction</h1>
    <form method="POST">
        {% for f in fields %}
        <label>{{ f }}</label><br>
        <input name="{{ f }}" type="number" step="any" required><br>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>
    {% if result %}
        <h2>Prediction: {{ result }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    result = None

    if request.method == "POST":
        user_input = [float(request.form[f]) for f in fields]
        scaled_input = scaler.transform([user_input])
        pred = model.predict(scaled_input)[0]
        result = "Positive (Diabetic)" if pred == 1 else "Negative (Healthy)"
    return render_template_string(HTML_PAGE, fields=fields, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
