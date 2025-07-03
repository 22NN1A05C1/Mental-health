
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load pre-trained model
model = joblib.load('mental_health_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        result = 'Low Risk'
    elif prediction[0] == 1:
        result = 'Moderate Risk'
    else:
        result = 'High Risk'

    return render_template('index.html', prediction_text=f'Mental Health Risk Level: {result}')

if __name__ == '__main__':
    app.run(debug=True)
