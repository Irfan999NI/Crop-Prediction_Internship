from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
sc = pickle.load(open("standscaler.pkl", "rb"))
mx = pickle.load(open("minmaxscaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html", result="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Convert input to FLOAT
        N = float(request.form["Nitrogen"])
        P = float(request.form["Phosporus"])
        K = float(request.form["Potassium"])
        temp = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"])
        ph = float(request.form["pH"])
        rainfall = float(request.form["Rainfall"])

        # ✅ Correct order
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
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

    except Exception as e:
        result = "❌ Error in prediction. Please check inputs."

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
