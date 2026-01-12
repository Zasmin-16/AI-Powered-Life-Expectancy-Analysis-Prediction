from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "app", "model", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "app", "model", "scaler.pkl"))

# ---------------- LOAD & CLEAN DATA ----------------
df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "clean_life_expectancy.csv"))

# 🔥 IMPORTANT FIX: clean column names
df.columns = df.columns.str.strip()

# Country encoding
country_map = {c: i for i, c in enumerate(sorted(df["Country"].unique()))}

# ---------------- DASHBOARD DATA ----------------
def get_dashboard_data():
    status_grp = df.groupby("Status")["Life expectancy"].mean()
    year_grp = df.groupby("Year")["Life expectancy"].mean()
    scatter_sample = df.sample(200, random_state=42)
    status_count = df["Status"].value_counts()

    donut_data = {
        "Alcohol": df["Alcohol"].mean(),
        "BMI": df["BMI"].mean(),
        "Schooling": df["Schooling"].mean(),
        "GDP": df["GDP"].mean()
    }

    radar_data = {
        "Schooling": df["Schooling"].mean(),
        "GDP": df["GDP"].mean(),
        "BMI": df["BMI"].mean(),
        "Alcohol": df["Alcohol"].mean(),
        "Total expenditure": df["Total expenditure"].mean()
    }

    return {
        "status_labels": status_grp.index.tolist(),
        "status_values": status_grp.values.tolist(),

        "year_labels": year_grp.index.astype(str).tolist(),
        "year_values": year_grp.values.tolist(),

        "gdp_life": list(zip(
            scatter_sample["GDP"].tolist(),
            scatter_sample["Life expectancy"].tolist()
        )),

        "pie_labels": status_count.index.tolist(),
        "pie_values": status_count.values.tolist(),

        "donut_labels": list(donut_data.keys()),
        "donut_values": list(donut_data.values()),

        "radar_labels": list(radar_data.keys()),
        "radar_values": list(radar_data.values())
    }

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    chart_data = get_dashboard_data()
    return render_template("dashboard.html", chart_data=chart_data)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        country_encoded = country_map[request.form["Country"]]
        status = int(request.form["Status"])

        features = [
            country_encoded,
            int(request.form["Year"]),
            status,
            float(request.form["Adult Mortality"]),
            float(request.form["infant deaths"]),
            float(request.form["Alcohol"]),
            float(request.form["percentage expenditure"]),
            float(request.form["Hepatitis B"]),
            float(request.form["Measles"]),
            float(request.form["BMI"]),
            float(request.form["under-five deaths"]),
            float(request.form["Polio"]),
            float(request.form["Total expenditure"]),
            float(request.form["Diphtheria"]),
            float(request.form["HIV/AIDS"]),
            float(request.form["GDP"]),
            float(request.form["Population"]),
            float(request.form["thinness 1-19 years"]),
            float(request.form["thinness 5-9 years"]),
            float(request.form["Income composition of resources"]),
            float(request.form["Schooling"])
        ]

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

    return render_template(
        "predict.html",
        prediction=prediction,
        countries=sorted(country_map.keys())
    )

if __name__ == "__main__":
    app.run(debug=True)
