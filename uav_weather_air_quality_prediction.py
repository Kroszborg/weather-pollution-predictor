import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data from CSV file
print("Loading data from CSV file...")
try:
    data = pd.read_csv('uav_sensor_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'uav_sensor_data.csv' was not found.")
    print("Please ensure the file is in the same directory as this script.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: The file 'uav_sensor_data.csv' is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: Unable to parse 'uav_sensor_data.csv'. Please check the file format.")
    exit(1)

# Display basic information about the dataset
print("\nDataset Information:")
print(data.info())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Handle datetime column
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
else:
    print("Warning: 'timestamp' column not found. Using index as timestamp.")
    data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data))

# Separate datetime and numerical columns
datetime_columns = ['timestamp']
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Impute missing values for numerical columns
numerical_imputer = SimpleImputer(strategy='mean')
data_imputed_numerical = pd.DataFrame(numerical_imputer.fit_transform(data[numerical_columns]), 
                                      columns=numerical_columns)

# Combine imputed numerical data with datetime column
data_imputed = pd.concat([data[datetime_columns], data_imputed_numerical], axis=1)

# Select features and target variables
feature_columns = ['temperature', 'humidity']
target_columns = ['rainfall', 'pm2.5', 'pm10', 'carbon_monoxide', 'air_quality_index']

# Ensure all required columns are present
missing_columns = [col for col in feature_columns + target_columns if col not in data_imputed.columns]
if missing_columns:
    print(f"Error: The following columns are missing from the dataset: {missing_columns}")
    print("Please ensure your CSV file contains all required columns.")
    exit(1)

X = data_imputed[feature_columns]
y = data_imputed[target_columns]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the multi-output regression model
print("\nTraining the model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
multi_output_rf = MultiOutputRegressor(rf_model)
multi_output_rf.fit(X_train_scaled, y_train)
print("Model training completed.")

# Make predictions
y_pred = multi_output_rf.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

# Print evaluation metrics
print("\nEvaluation Metrics:")
for i, col in enumerate(y.columns):
    print(f"{col}:")
    print(f"  RMSE: {rmse[i]:.2f}")
    print(f"  R2 Score: {r2[i]:.2f}")
    print(f"  MAE: {mae[i]:.2f}")

# Visualize predictions vs actual values
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle("Predictions vs Actual Values")

for i, (ax, col) in enumerate(zip(axes.ravel(), y.columns)):
    ax.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5)
    ax.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
            [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
            'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(col)

plt.tight_layout()
plt.savefig('predictions_vs_actual.png')
print("Predictions vs Actual Values plot saved as 'predictions_vs_actual.png'")

# Feature importance analysis
feature_importance = np.mean([tree.feature_importances_ for tree in multi_output_rf.estimators_], axis=0)
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature Importance plot saved as 'feature_importance.png'")

# Time series visualization
plt.figure(figsize=(12, 8))
for col in y.columns:
    plt.plot(data_imputed['timestamp'], data_imputed[col], label=col)
plt.title("Time Series of Target Variables")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig('time_series.png')
print("Time Series plot saved as 'time_series.png'")

print("\nScript execution completed.")
