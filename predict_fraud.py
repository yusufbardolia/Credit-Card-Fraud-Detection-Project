import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Step 1: Load the Trained Model
model = joblib.load('credit_fraud_detection_model.pkl')
print("Model loaded successfully.")

# Step 2: Create a Sample DataFrame with All Required Columns
# Assuming your model was trained with columns V1 to V28, scaled_amount, and scaled_time

# Mock data for demonstration (you would replace this with actual new data)
new_data = pd.DataFrame({
    'V1': np.random.randn(4),
    'V2': np.random.randn(4),
    'V3': np.random.randn(4),
    'V4': np.random.randn(4),
    'V5': np.random.randn(4),
    'V6': np.random.randn(4),
    'V7': np.random.randn(4),
    'V8': np.random.randn(4),
    'V9': np.random.randn(4),
    'V10': np.random.randn(4),
    'V11': np.random.randn(4),
    'V12': np.random.randn(4),
    'V13': np.random.randn(4),
    'V14': np.random.randn(4),
    'V15': np.random.randn(4),
    'V16': np.random.randn(4),
    'V17': np.random.randn(4),
    'V18': np.random.randn(4),
    'V19': np.random.randn(4),
    'V20': np.random.randn(4),
    'V21': np.random.randn(4),
    'V22': np.random.randn(4),
    'V23': np.random.randn(4),
    'V24': np.random.randn(4),
    'V25': np.random.randn(4),
    'V26': np.random.randn(4),
    'V27': np.random.randn(4),
    'V28': np.random.randn(4),
    'Amount': [100.0, 250.0, 15.0, 75.0],
    'Time': [50000, 60000, 70000, 80000]
})

# Feature Scaling for 'Amount' and 'Time'
scaler = StandardScaler()
new_data['scaled_amount'] = scaler.fit_transform(new_data['Amount'].values.reshape(-1, 1))
new_data['scaled_time'] = scaler.fit_transform(new_data['Time'].values.reshape(-1, 1))
new_data = new_data.drop(['Amount', 'Time'], axis=1)

print("New data prepared with all required features.")

# Step 3: Make Predictions
predictions = model.predict(new_data)
print("Predictions complete.")

# Output the predictions
print("Predictions:", predictions)

# Save the predictions to a CSV file
new_data['Prediction'] = predictions
new_data.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv.")
