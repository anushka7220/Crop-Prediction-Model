import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Create flask app
flask_app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.pickle')
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please ensure the file exists.")
    model = None


@flask_app.route("/")
def home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", prediction_text="Model not found.")

    try:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction[0]))
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error in prediction: {str(e)}")


if __name__ == "__main__":
    flask_app.run(debug=True)
