from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# -------------------- TRAIN MODEL ON THE FLY --------------------
print("⚙️ Training model (no model.pkl needed)...")
data = pd.read_csv("diabetes.csv", encoding='latin1')
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)
print("✅ Model trained successfully!")

# -------------------- SIMPLE HTML USER INTERFACE --------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🩺 Diabetes Prediction</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: linear-gradient(to right, #e0f7fa, #e1bee7);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px;
    }
    h1 { color: #333; }
    form {
      background: white;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 0 15px rgba(0,0,0,0.2);
      width: 320px;
      text-align: center;
    }
    input {
      width: 90%;
      padding: 8px;
      margin: 6px 0;
      b
