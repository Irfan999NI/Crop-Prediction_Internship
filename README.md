# Crop Recommendation System

A Flask-based machine learning application that recommends the most suitable crop for a field using soil and climate data such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

## Overview

This project helps farmers and agricultural planners make faster crop decisions by analyzing agronomic parameters and predicting the best crop match from a trained machine learning model.

The application is built with:

- Python
- Flask
- NumPy
- scikit-learn
- HTML/CSS/Bootstrap for the frontend

## Features

- Input soil nutrient values and weather conditions
- Predict suitable crop from trained ML model
- Assess current nitrogen, phosphorus, and potassium soil fertility
- Suggest fertilizer sources and crop-specific nutrient focus
- Reject field values outside the training dataset's range instead of extrapolating a crop prediction
- Simple web interface for easy field data entry
- Works with common crop categories such as rice, maize, cotton, mango, banana, grapes, and more
- Useful for prototype agricultural decision-support systems

## Tech Stack

- Backend: Flask
- ML model: scikit-learn
- Data preprocessing: StandardScaler and MinMaxScaler
- Frontend: HTML, CSS, Bootstrap

## Project Structure

```text
Crop-Recommendation-system/
├── app.py                 # Flask web application
├── model.pkl              # Trained crop prediction model
├── minmaxscaler.pkl       # MinMaxScaler used during preprocessing
├── standscaler.pkl        # StandardScaler used during preprocessing
├── Crop_recommendation.csv
│                         # Dataset used for model training
├── Crop Recommendation Using Machine Learning.ipynb
│                         # Notebook for model experimentation
├── requirements.txt        # Python dependencies
├── Procfile                # Production web-server command for PaaS hosts
├── .gitignore
├── static/
│   └── css/
│       └── Sytle.css
├── templates/
│   ├── index.html         # Main prediction form
│   └── login.html         # Additional template
├── tests/
│   └── test_app.py        # Application smoke tests
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Irfan999NI/Crop-Prediction_Internship.git
cd Crop-Prediction_Internship
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
```

On Windows:

```powershell
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Flask server:

```bash
python app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://127.0.0.1:5000/
```

## Deploying to a hosting provider

The repository includes a `Procfile` that starts the app with Gunicorn, a production WSGI server. For a Linux-based Python hosting provider:

1. Connect the repository and use `pip install -r requirements.txt` as the build/install command.
2. Use `gunicorn --bind 0.0.0.0:$PORT --workers 1 --access-logfile - --error-logfile - app:app` as the start command if the provider does not read the `Procfile` automatically.
3. Keep `app.py`, `model.pkl`, `standscaler.pkl`, and `minmaxscaler.pkl` in the deployed repository root.
4. Configure the provider's health check to request `/health` (optional).

The app loads model artifacts relative to `app.py`, so startup does not depend on the hosting process's working directory. Do not start the production deployment with `python app.py`; that command is intended for local development.

Run the app checks locally with:

```bash
python -m unittest discover -s tests
```

## How It Works

The app accepts field data such as:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

These values are processed using the saved scalers and passed to the trained model, which predicts the best crop.

## Example

Example input values:

- Nitrogen: 90
- Phosphorus: 42
- Potassium: 43
- Temperature: 20-30°C
- Humidity: 60-80%
- pH: 6-7
- Rainfall: 100-200 mm

The model will output a recommended crop such as Rice, Maize, Banana, or another suitable crop label based on the trained dataset.

## Notes

- This is a learning/demo project and is best suited for educational use.
- Recommendations should be treated as guidance and validated with agronomic expertise or local agricultural advice.
- The model depends on the quality and coverage of the training dataset.

## Future Improvements

- Add crop yield prediction
- Add user authentication and saved recommendations
- Improve the UI and add charts/visualizations
- Expand the dataset and retrain the model with more crop classes and region-specific data

## License

This project does not currently include a license file. Please check the repository for usage terms before commercial deployment.
