from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "app", "model", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "app", "model", "scaler.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

    return render_template("predict.html", prediction=prediction)
