import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Config
MODEL_PATH = os.environ.get('MODEL_PATH', 'model_pipeline.pkl')
FEATURES = ['acousticness', 'speechiness', 'key', 'liveness', 'mode']

# App
app = Flask(__name__, template_folder='templates')

# Lazy-loaded model
_model = None


def load_model(path=MODEL_PATH):
    """Load and cache the model pipeline saved with joblib."""
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at '{path}'.Train the model and save pipeline to this path."
            )
        _model = joblib.load(path)
    return _model


def validate_and_prepare_input(input_map):
  
    row = {f: input_map.get(f, None) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    X = X.apply(pd.to_numeric, errors='coerce')
    return X


def predict_single(input_map, model=None):
   
    if model is None:
        model = load_model()
    X = validate_and_prepare_input(input_map)
    pred = model.predict(X)
    return float(np.squeeze(pred))


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            # read values from form; allow empty fields to be sent as blank
            input_data = {
                'acousticness': request.form.get('acousticness', type=float),
                'speechiness': request.form.get('speechiness', type=float),
                'key': request.form.get('key', type=float),
                'liveness': request.form.get('liveness', type=float),
                'mode': request.form.get('mode', type=float),
            }
            pred = predict_single(input_data, model=load_model())
            prediction = round(pred, 3)
        except Exception as e:
            error = str(e)
    return render_template('index.html', prediction=prediction, error=error)


@app.route('/api/predict', methods=['POST'])
def api_predict():

    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body must be an object/dict of feature values."}), 400
        # Only pick expected features (ignore extras)
        input_map = {f: payload.get(f, None) for f in FEATURES}
        pred = predict_single(input_map, model=load_model())
        return jsonify({"prediction": float(pred)})
    except FileNotFoundError as fnfe:
        return jsonify({"error": str(fnfe)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Helpful message if model is missing when launching directly
    try:
        load_model()
    except FileNotFoundError:
        print(f"Warning: model file '{MODEL_PATH}' not found. Start the app after creating the model pickle.")
    # Run dev server
    app.run(host='0.0.0.0', port=5000, debug=True)