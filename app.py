import pickle
from pathlib import Path

import numpy as np
from flask import Flask, redirect, request, render_template, url_for

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
FEATURE_NAMES = ("Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall")


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


def nutrient_status(value, low_limit, medium_limit):
    if value < low_limit:
        return "Low"
    if value < medium_limit:
        return "Medium"
    return "High"


def get_fertility_recommendation(nitrogen, phosphorus, potassium, ph):
    nitrogen_status = nutrient_status(nitrogen, 40, 80)
    phosphorus_status = nutrient_status(phosphorus, 20, 40)
    potassium_status = nutrient_status(potassium, 30, 60)

    statuses = [nitrogen_status, phosphorus_status, potassium_status]
    if "Low" in statuses:
        overall = "Low"
    elif "Medium" in statuses:
        overall = "Medium"
    else:
        overall = "Good"

    fertilizer_steps = []
    if nitrogen_status == "Low":
        fertilizer_steps.append("Nitrogen: apply a nitrogen source such as urea or ammonium sulfate; split applications where practical.")
    elif nitrogen_status == "Medium":
        fertilizer_steps.append("Nitrogen: use a moderate nitrogen application and split it according to crop growth stages.")

    if phosphorus_status == "Low":
        fertilizer_steps.append("Phosphorus: use DAP, SSP, or rock phosphate; incorporate it near the root zone.")
    elif phosphorus_status == "Medium":
        fertilizer_steps.append("Phosphorus: maintain supply with DAP or SSP as recommended by a soil test.")

    if potassium_status == "Low":
        fertilizer_steps.append("Potassium: use Muriate of Potash (MOP) or sulfate of potash, especially for fruit and tuber crops.")
    elif potassium_status == "Medium":
        fertilizer_steps.append("Potassium: maintain with MOP or sulfate of potash during later growth stages.")

    if not fertilizer_steps:
        fertilizer_steps.append("Maintain fertility with well-decomposed compost or farmyard manure and a balanced NPK product.")
    else:
        fertilizer_steps.append("Add well-decomposed compost or farmyard manure to improve soil structure and nutrient retention.")

    if ph < 5.5:
        fertilizer_steps.append("Soil pH: the soil is acidic; test for lime requirement before applying agricultural lime.")
    elif ph > 7.5:
        fertilizer_steps.append("Soil pH: the soil is alkaline; add organic matter and consider sulfur or a crop suited to alkaline soils.")
    else:
        fertilizer_steps.append("Soil pH: the pH is in the generally usable range for many crops.")

    return {
        "overall": overall,
        "nitrogen": nitrogen_status,
        "phosphorus": phosphorus_status,
        "potassium": potassium_status,
        "steps": fertilizer_steps,
    }


CROP_FERTILIZER_FOCUS = {
    "Rice": "Use balanced nitrogen in split doses, with phosphorus at basal application.",
    "Maize": "Use nitrogen in split doses; supply phosphorus at planting and potassium at peak growth.",
    "Cotton": "Prioritize phosphorus during establishment and split nitrogen and potassium applications.",
    "Sugarcane": "Use a higher nitrogen and potassium schedule in divided applications; do not skip phosphorus.",
    "Coconut": "Use organic manure plus a potassium-rich balanced NPK mixture.",
    "Apple": "Use split nitrogen applications and adequate potassium and phosphorus; avoid excess nitrogen.",
    "Grapes": "Use balanced NPK with emphasis on potassium and organic matter.",
    "Banana": "Use frequent, split potassium applications plus nitrogen and phosphorus.",
    "Pomegranate": "Use moderate nitrogen, adequate phosphorus, and potassium during flowering and fruiting.",
    "Mango": "Use mature compost and balanced NPK; reduce nitrogen after establishment.",
}

@app.route("/")
def index():
    return render_template(
        "index.html",
        result=request.args.get("result", ""),
        fertility=request.args.get("fertility", ""),
        fertilizer=request.args.get("fertilizer", ""),
        soil_steps=request.args.getlist("soil_step"),
        crop_focus=request.args.get("crop_focus", ""),
        values={
            "Nitrogen": request.args.get("N", ""),
            "Phosphorus": request.args.get("P", ""),
            "Potassium": request.args.get("K", ""),
            "Temperature": request.args.get("T", ""),
            "Humidity": request.args.get("H", ""),
            "pH": request.args.get("pH", ""),
            "Rainfall": request.args.get("R", ""),
        },
    )


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))

    try:
        nitrogen = parse_numeric_field("Nitrogen")
        phosphorus = parse_numeric_field("Phosphorus", "Phosporus")
        potassium = parse_numeric_field("Potassium")
        temperature = parse_numeric_field("Temperature")
        humidity = parse_numeric_field("Humidity")
        ph = parse_numeric_field("pH")
        rainfall = parse_numeric_field("Rainfall")
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    except ValueError as error:
        return redirect(url_for("index", result=f"❌ {error}"))

    if nitrogen < 0 or phosphorus < 0 or potassium < 0 or temperature < 0 or rainfall < 0:
        return redirect(url_for("index", result="❌ Soil nutrient and weather values cannot be negative."))
    if not 0 <= humidity <= 100:
        return redirect(url_for("index", result="❌ Humidity must be between 0 and 100%."))
    if not 0 <= ph <= 14:
        return redirect(url_for("index", result="❌ pH must be between 0 and 14."))

    for name, value, minimum, maximum in zip(
        FEATURE_NAMES, features[0], mx.data_min_, mx.data_max_
    ):
        if not minimum <= value <= maximum:
            return redirect(url_for(
                "index",
                result=f"❌ {name} must be between {minimum:g} and {maximum:g} (the model's training-data range).",
            ))

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

    crop = crop_dict.get(prediction, "Unknown")
    fertility = get_fertility_recommendation(nitrogen, phosphorus, potassium, ph)
    crop_focus = CROP_FERTILIZER_FOCUS.get(
        crop,
        "Use balanced NPK fertilizer and organic manure according to a local soil test.",
    )

    return redirect(url_for(
        "index",
        result=crop,
        fertility=f"{fertility['overall']} fertility",
        fertilizer=f"N: {fertility['nitrogen']} | P: {fertility['phosphorus']} | K: {fertility['potassium']}",
        soil_step=fertility["steps"],
        crop_focus=crop_focus,
        N=nitrogen,
        P=phosphorus,
        K=potassium,
        T=temperature,
        H=humidity,
        pH=ph,
        R=rainfall,
    ))


if __name__ == "__main__":
    app.run()
