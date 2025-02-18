from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the XGBoost model
with open('models/xgbrf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example preprocessing steps (you may need to adjust based on your actual preprocessing)
def preprocess_input(data):
    # Convert categorical variables to one-hot encoding
    cat_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    data = pd.get_dummies(data, columns=cat_columns)
    
    # Ensure all columns in the model's training data are present, filling with 0 if not
    model_columns = model.get_booster().feature_names
    for col in model_columns:
        if col not in data.columns:
            data[col] = 0
    
    # Reorder columns to match the model's feature order
    data = data[model_columns]
    
    return data

@app.route('/')
def home():
    return render_template('index.html', prediction_text='', 
                           item_weight='', item_fat_content='', item_visibility='', 
                           item_type='', item_mrp='', outlet_identifier='', outlet_establishment_year='', 
                           outlet_size='', outlet_location_type='', outlet_type='')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from form
    item_weight = float(request.form['item_weight'])
    item_fat_content = request.form['item_fat_content']
    item_visibility = float(request.form['item_visibility'])
    item_type = request.form['item_type']
    item_mrp = float(request.form['item_mrp'])
    outlet_identifier = request.form['outlet_identifier']
    outlet_establishment_year = int(request.form['outlet_establishment_year'])
    outlet_size = request.form['outlet_size']
    outlet_location_type = request.form['outlet_location_type']
    outlet_type = request.form['outlet_type']

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Item_Weight': [item_weight],
        'Item_Fat_Content': [item_fat_content],
        'Item_Visibility': [item_visibility],
        'Item_Type': [item_type],
        'Item_MRP': [item_mrp],
        'Outlet_Establishment_Year': [outlet_establishment_year],
        'Outlet_Size': [outlet_size],
        'Outlet_Location_Type': [outlet_location_type],
        'Outlet_Type': [outlet_type],
        'Outlet_Identifier': [outlet_identifier]
    })

    # Perform preprocessing on the input data
    input_data_preprocessed = preprocess_input(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_data_preprocessed.values.reshape(1, -1))[0]

    # Calculate prediction interval (example using standard deviation of residuals)
    # You may need to adjust this based on your specific model and data
    # For simplicity, assume the standard deviation of residuals from your test set
    residual_std = 1024.236221990072  # Replace with actual standard deviation of residuals

    # Define confidence level (e.g., 95% confidence interval)
    confidence_level = 0.95

    # Calculate margin of error (z-score for 95% confidence interval is approximately 1.96)
    margin_of_error = 1.96 * residual_std

    # Prediction interval
    lower_bound = prediction - margin_of_error
    upper_bound = prediction + margin_of_error

    # Render the template with input values and prediction results
    return render_template('index.html', 
                           prediction_text=f'Predicted Sales: {prediction:.2f}',
                           prediction_range_text=f'Sales between: {lower_bound:.2f} to {upper_bound:.2f}',
                           item_weight=item_weight,
                           item_fat_content=item_fat_content,
                           item_visibility=item_visibility,
                           item_type=item_type,
                           item_mrp=item_mrp,
                           outlet_identifier=outlet_identifier,
                           outlet_establishment_year=outlet_establishment_year,
                           outlet_size=outlet_size,
                           outlet_location_type=outlet_location_type,
                           outlet_type=outlet_type)

if __name__ == '__main__':
    app.run(debug=True)

