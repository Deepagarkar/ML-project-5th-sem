# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# load dataset
df = pd.read_csv("diabetes.csv")

# Example: adjust columns if your csv columns differ
# Typical Pima dataset columns:
# ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pipeline: scaler + RF
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe.fit(X_train, y_train)

# Save model
joblib.dump(pipe, "model.pkl")
print("Model trained and saved to model.pkl")
