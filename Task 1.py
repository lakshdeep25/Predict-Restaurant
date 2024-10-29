import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
data = pd.read_csv('Dataset.csv')

# 1. Check for NaN values in the raw dataset
print("NaN values in the dataset before preprocessing:")
print(data.isna().sum())

# Handle missing values (forward fill and then fill remaining NaN with 0)
data.fillna(method='ffill', inplace=True)
data.fillna(0, inplace=True)  # Ensures no NaN values remain

# 2. Check for NaN values after filling missing values
print("\nNaN values after filling missing values:")
print(data.isna().sum())

# Encoding categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['City', 'Cuisines', 'Has Table booking', 'Has Online delivery', 
                    'Is delivering now', 'Switch to order menu', 'Currency']

for col in categorical_cols:
    if col in data.columns:
        data[col] = label_encoder.fit_transform(data[col].astype(str))

# 3. Check for NaN values after encoding categorical columns
print("\nNaN values after encoding categorical columns:")
print(data.isna().sum())

# Defining features (X) and target (y)
X = data.drop('Aggregate rating', axis=1)  # Features
y = data['Aggregate rating']  # Target variable

# 4. Check for NaN values in features (X) and target (y)
print("\nNaN values in features (X) and target (y) after preprocessing:")
print("X NaN values:", X.isna().sum().sum())
print("y NaN values:", y.isna().sum())

# If there are still NaN values in features or target, handle them (drop them or replace them)
X.fillna(0, inplace=True)  # Replace any lingering NaN values in features
y.fillna(y.median(), inplace=True)  # Replace NaN target values with median

# Splitting the dataset into numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
X_numerical = X[numerical_cols]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_numerical, y, test_size=0.2, random_state=42)

# Scaling numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a regression model (Linear Regression in this case)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Convert predictions to a Pandas Series for easier handling
y_pred_series = pd.Series(y_pred, index=X_test.index)

# 5. Check if any NaN values in predictions
print("\nNaN values in predictions:")
print(y_pred_series.isna().sum())

# If there are NaN values in predictions, handle them (e.g., replace with median of predictions)
if y_pred_series.isna().sum() > 0:
    y_pred_series.fillna(y_pred_series.median(), inplace=True)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_series)
r2 = r2_score(y_test, y_pred_series)

print(f'\nMean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Add predictions back to the original DataFrame
data.loc[X_test.index, 'Predicted rating'] = y_pred_series

# Optionally, save the updated DataFrame to a new CSV file
data.to_csv('updated_restaurant_data_with_predictions.csv', index=False)

# Display the DataFrame with predictions
print("\nFirst 10 entries of actual and predicted ratings:")
print(data[['Aggregate rating', 'Predicted rating']].head(10))
