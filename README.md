# UAV Sensor Data Analysis and Prediction Model

This project contains a Python script for analyzing and predicting various environmental factors based on UAV (Unmanned Aerial Vehicle) sensor data. The script processes data including temperature, humidity, rainfall, particulate matter (PM2.5 and PM10), carbon monoxide levels, and air quality index.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Input Data Format](#input-data-format)
5. [Script Overview](#script-overview)
6. [Outputs](#outputs)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

1. Clone this repository or download the script file.
2. Install the required Python packages:

   ```
   pip install numpy pandas matplotlib scikit-learn
   ```

## Usage

1. Ensure your UAV sensor data is in a CSV file named `uav_sensor_data.csv` in the same directory as the script.
2. Run the script:

   ```
   python uav_weather_air_quality_prediction.py
   ```

3. The script will process the data, train the model, and generate output files.

## Input Data Format

The `uav_sensor_data.csv` file should contain the following columns:

- timestamp
- temperature
- humidity
- rainfall
- pm2.5
- pm10
- carbon_monoxide
- air_quality_index

Ensure that your CSV file has a header row with these column names.

## Script Overview

The script performs the following operations:

1. Loads and preprocesses the UAV sensor data.
2. Handles missing values using mean imputation.
3. Splits the data into training and testing sets.
4. Trains a Random Forest Regressor model for multi-output regression.
5. Evaluates the model's performance using various metrics.
6. Generates visualizations for predictions, feature importance, and time series data.

## Outputs

The script produces the following outputs:

1. Console output:
   - Dataset information
   - Missing value counts
   - Model training progress
   - Evaluation metrics (RMSE, R2 Score, MAE) for each target variable

2. Visualization files:
   - `predictions_vs_actual.png`: Scatter plots comparing predicted vs actual values for each target variable.
   - `feature_importance.png`: Bar chart showing the importance of input features.
   - `time_series.png`: Line plot showing the time series of all target variables.

## Customization

To customize the script for your specific needs:

1. Modify the `feature_columns` and `target_columns` lists to change which variables are used as inputs and outputs.
2. Adjust the `RandomForestRegressor` parameters in the `rf_model` initialization to tune the model.
3. Add or modify visualizations in the plotting sections to create custom charts.

## Troubleshooting

- If you encounter a "File not found" error, ensure that `uav_sensor_data.csv` is in the same directory as the script.
- For "Column not found" errors, check that your CSV file contains all the required columns listed in the [Input Data Format](#input-data-format) section.
- If you get unexpected results, verify that your data is in the correct format and that there are no inconsistencies in the measurements.

For any other issues or questions, please open an issue in the GitHub repository.
