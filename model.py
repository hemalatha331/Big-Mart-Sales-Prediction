import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load data
data = pd.read_csv('dataset.csv')

# Drop unnecessary columns
data = data.drop(['Item_Identifier'], axis=1)

# Prepare X and y
X = data.drop(['Item_Outlet_Sales'], axis=1)  # Features
y = data['Item_Outlet_Sales']  # Target variable

# Include Outlet_Identifier in categorical columns for one-hot encoding
cat_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
data_processed = pd.get_dummies(X, columns=cat_columns)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_processed, y, test_size=0.2, random_state=42)

# Initialize model
model = XGBRFRegressor(n_estimators=100, max_depth=6, objective='reg:squarederror', subsample=0.8, colsample_bynode=0.8)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Save model (optional)
with open('xgbrf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

