from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import pickle

app = Flask(__name__)

# Load the trained XGBoost model from the pickle file
with open('irrigation.pkl', 'rb') as model_file:
    loaded_regressor = pickle.load(model_file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    features = [request.form.get('temperature'), request.form.get('pressure'), request.form.get('altitude'), request.form.get('soilmoisture')]


    # Check for None values and handle them
    if any(feature is None or feature == '' for feature in features):
        return render_template('index.html', prediction="Please provide values for all features")

    # Print features and their values for debugging
    for i, feature in enumerate(features, start=1):
        print(f"Feature col{i}: {feature}")

    # Convert features to float, explicitly filtering out None values
    try:
        float_features = [float(feature) for feature in features if feature is not None and feature != '']
    except ValueError as e:
        print(f"Error converting input to float: {e}")
        return render_template('index.html', prediction="Invalid input. Please provide valid numeric values.")

    # Check if all features are valid
    if len(float_features) != 4:
        return render_template('index.html', prediction="Please provide values for all features")

    print("Float features:", float_features)  # Added for debugging

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([float_features], columns=['temperature', 'pressure', 'altitude', 'soilmiosture'])

    # Use the loaded model to make predictions
    prediction = loaded_regressor.predict(input_data)

    return render_template('index.html', prediction=f"The predicted value is: {prediction[0]}")

if __name__ == '__main__':
    app.run(port=5000,debug=True, threaded=True)
