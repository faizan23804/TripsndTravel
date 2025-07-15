from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load("random_forest.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Define feature names
categorical_cols = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation'
]

numeric_cols = [
    'Age', 'CityTier', 'DuraonOfPitch', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'MonthlyIncome', 'Total_Visitors'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define categorical and numerical features in exact order
        cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
        num_cols = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfFollowups',
                    'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
                    'PitchSatisfactionScore', 'OwnCar', 'MonthlyIncome', 'Total_Visitors']
        all_cols = cat_cols + num_cols

        # Collect form data
        form = request.form
        cat_data = [form[col] for col in cat_cols]
        num_data = [float(form[col]) for col in num_cols]
        input_data = cat_data + num_data

        # Create a DataFrame with correct column order
        input_df = pd.DataFrame([input_data], columns=all_cols)

        # Load preprocessor and model
        preprocessor = joblib.load("preprocessor.pkl")
        model = joblib.load("random_forest.pkl")

        # Transform input and predict
        X_transformed = preprocessor.transform(input_df)
        prediction = model.predict(X_transformed)[0]

        # Format the result
        result = "✅ Likely to purchase the Wellness Package." if prediction == 1 else "❌ Unlikely to purchase the Wellness Package."

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"❗ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
