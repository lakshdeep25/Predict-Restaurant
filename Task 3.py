import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and inspect the dataset
data = pd.read_csv('Dataset.csv')

# Step 2: Data Cleaning and Preprocessing
# Dropping rows with missing values in key columns
data_cleaned = data.dropna(subset=['Cuisines', 'Average Cost for two', 'Has Table booking', 'Has Online delivery'])

# Mapping categorical variables (Yes/No) to numerical values (1/0)
data_cleaned['Has Table booking'] = data_cleaned['Has Table booking'].map({'Yes': 1, 'No': 0})
data_cleaned['Has Online delivery'] = data_cleaned['Has Online delivery'].map({'Yes': 1, 'No': 0})

# Handling multi-label cuisine (using the first cuisine for simplicity, can be improved later)
data_cleaned['Cuisines'] = data_cleaned['Cuisines'].apply(lambda x: x.split(',')[0])

# Encoding the 'Cuisines' column using one-hot encoding
data_encoded = pd.get_dummies(data_cleaned, columns=['Cuisines'])

# Step 3: Defining Features and Labels
X = data_encoded.drop(columns=['Restaurant Name', 'Cuisines'])  # Drop non-feature columns
y = data_cleaned['Cuisines']  # Label is the Cuisine

# Step 4: Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Building and Training the Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Making Predictions
y_pred = rf_model.predict(X_test)

# Step 7: Evaluating the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

