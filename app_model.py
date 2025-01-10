from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

root_path ="/home/Estebanson/Taller_despliegue_directo/"


app = Flask(__name__)
app.config['DEBUG'] = False

# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def hello():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Revenue Prediction - Retro Style</title>
    <style>
        body {
            background-color: #2c3e50;
            color: #ecf0f1;
            font-family: 'Courier New', Courier, monospace;
            text-align: center;
            margin: 0;
            padding: 0;
        }

        .container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        h1 {
            font-size: 3rem;
            color: #e74c3c;
            text-shadow: 2px 2px 4px #000000;
        }

        p {
            font-size: 1.2rem;
            margin: 20px 0;
        }

        .button {
            background-color: #16a085;
            color: #ecf0f1;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }

        .button:hover {
            background-color: #1abc9c;
            box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.3);
            transform: scale(1.05);
        }

        .footer {
            position: absolute;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 0.9rem;
            color: #bdc3c7;
        }

        .neon-text {
            font-family: 'Pacifico', cursive;
            font-size: 2.5rem;
            color: #f39c12;
            text-shadow: 0 0 5px #f1c40f, 0 0 10px #f39c12, 0 0 20px #f39c12, 0 0 30px #e67e22, 0 0 40px #d35400;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="neon-text">Welcome to Revenue Predictor</h1>
        <p>Your retro-style tool to forecast revenue based on TV, Radio, and Newspaper investments.</p>
        <button class="button" onclick="alert('Let\'s predict some revenue!')">Get Started</button>
    </div>
    <div class="footer">
        <p>&copy; 2025 Revenue Predictor | Designed with <span style="color: #e74c3c;">♥</span> in Retro Style</p>
    </div>
</body>
</html>
'''

# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open(root_path + 'ad_model.pkl', 'rb'))
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])

    return jsonify({'predictions': prediction[0]})

# Enruta la funcion al endpoint /api/v1/retrain
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv(root_path + 'data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        pickle.dump(model, open(root_path + 'ad_model.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

if __name__ == "__main__":
    app.run()