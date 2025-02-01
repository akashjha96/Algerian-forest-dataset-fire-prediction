import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load the models
regressor = pickle.load(open('models/regressor.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
ridge = pickle.load(open('models/ridge.pkl', 'rb'))

app = Flask(__name__)

# Route to render the form for input data
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        data = request.form.to_dict()
        # Convert form data into a list of features
        features = np.array([[float(data['Temperature']),
                             float(data['RH']),
                             float(data['Ws']),
                             float(data['Rain']),
                             float(data['FFMC']),
                             float(data['DMC']),
                             float(data['DC']),
                             float(data['ISI']),
                             float(data['BUI']),
                             float(data['FWI']),
                             int(data['Classes']),
                             int(data['Region'])]])

        # Scale the data
        features_scaled = scaler.transform(features)

        # Make predictions using the loaded models
        regressor_prediction = regressor.predict(features_scaled)
        ridge_prediction = ridge.predict(features_scaled)

        # Combine predictions in a response (optional)
        result = {
            'regressor_prediction': regressor_prediction[0],
            'ridge_prediction': ridge_prediction[0]
        }

        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
