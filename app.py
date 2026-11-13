import pickle
from pathlib import Path

import numpy as np
from flask import Flask, redirect, request, render_template, url_for

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


def load_artifact(filename):
    with (BASE_DIR / filename).open("rb") as artifact_file:
        return pickle.load(artifact_file)


model = load_artifact("model.pkl")
sc = load_artifact("standscaler.pkl")
mx = load_artifact("minmaxscaler.pkl")


def parse_numeric_field(name, legacy_name=None):
    value = request.form.get(name)
    if value is None and legacy_name:
        value = request.form.get(legacy_name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required.")

    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid number.") from error

    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number

@app.route("/")
def index():
    return render_template("index.html", result=request.args.get("result", ""))


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))

    try:
        features = np.array([[
            parse_numeric_field("Nitrogen"),
            parse_numeric_field("Phosphorus", "Phosporus"),
            parse_numeric_field("Potassium"),
            parse_numeric_field("Temperature"),
            parse_numeric_field("Humidity"),
            parse_numeric_field("pH"),
            parse_numeric_field("Rainfall"),
        ]])
    except ValueError as error:
        return redirect(url_for("index", result=f"❌ {error}"))

    mx_scaled = mx.transform(features)
    final_features = sc.transform(mx_scaled)
    prediction = model.predict(final_features)[0]

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
        10: "Watermelon", 11: "Grapes", 12: "Mango",
        13: "Banana", 14: "Pomegranate", 15: "Lentil",
        16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea",
        22: "Coffee"
    }

    result = f"🌾 Recommended Crop: {crop_dict.get(prediction, 'Unknown')}"

    return redirect(url_for("index", result=result))


if __name__ == "__main__":
    app.run()
