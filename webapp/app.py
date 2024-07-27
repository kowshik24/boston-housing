from flask import Flask, request, jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def scale(payload):
    scaler = StandardScaler()
    return scaler.fit_transform(payload)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Input Sample:
    {
        "CHAS": {"0": 0},"RM": {"0": 6.575},"TAX": {"0": 296.0},"PTRATIO": {"0": 15.3},"B": {"0": 396.9},"LSTAT": {"0": 4.98}

    }

    Output Sample:
    {
        "prediction": [20.35373177134412]
    }
    """

    clf = joblib.load("boston_housing_prediction.joblib")
    payload = pd.DataFrame(request.json)
    scaled_payload = scale(payload)
    prediction = list(clf.predict(scaled_payload))
    return jsonify({"prediction": prediction})


@app.route('/predict_house', methods=['POST'])
def predict_house():
    # Parse input data from the request
    data = request.form
    CHAS = float(data['CHAS'])
    RM = float(data['RM'])
    TAX = float(data['TAX'])
    PTRATIO = float(data['PTRATIO'])
    B = float(data['B'])
    LSTAT = float(data['LSTAT'])

    payload = pd.DataFrame({"CHAS": {"0": CHAS},"RM": {"0": RM},"TAX": {"0": TAX},"PTRATIO": {"0": PTRATIO},"B": {"0": B},"LSTAT": {"0": LSTAT}})
    scaled_payload = scale(payload)
    # Load the model and make a prediction
    clf = joblib.load("boston_housing_prediction.joblib")
    prediction = list(clf.predict(scaled_payload))


    # Return the result as a JSON response
    return jsonify({"Prediction": prediction[0]})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
