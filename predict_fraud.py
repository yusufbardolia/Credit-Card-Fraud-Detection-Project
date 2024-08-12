# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Load the model
model_filename = 'credit_fraud_detection_model.pkl'
rf_model = joblib.load(model_filename)

# Load the test dataset
test_data_path = r'c:\Users\yusuf\Downloads\archive (4)\fraudTest.csv'
test_data = pd.read_csv(test_data_path)

# Print column names to verify them
print("Column names in the test dataset:", test_data.columns)

# Convert categorical columns to numeric using the same encoders as for training
# Load or define label encoders if they are saved
label_encoders = {
    'merchant': LabelEncoder(),
    'category': LabelEncoder(),
    'gender': LabelEncoder(),
    'job': LabelEncoder()
}

# Ensure label encoders are fitted with the training data categories
for column, le in label_encoders.items():
    test_data[column] = le.fit_transform(test_data[column])

# Convert 'trans_date_trans_time' to UNIX timestamp
test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])
test_data['trans_date_trans_time'] = test_data['trans_date_trans_time'].astype(np.int64)  # Use UNIX timestamps

# Feature Scaling
scaler = StandardScaler()
test_data['scaled_amt'] = scaler.fit_transform(test_data[['amt']])
test_data['scaled_trans_time'] = scaler.fit_transform(test_data[['trans_date_trans_time']])

# Drop columns not used in modeling
test_data = test_data.drop(['amt', 'trans_date_trans_time', 'Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long'], axis=1)

# Prepare the features and target for testing
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud']

# Make Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Evaluate Performance
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_pred)}')

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkgreen', lw=2, label='Random Forest ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Random Forest')
plt.legend(loc="lower right")
plt.show()

# Show a sample of predictions with probabilities
test_data['Predicted_Label'] = y_pred
test_data['Predicted_Probability'] = y_pred_proba

# Print a sample of the results
print(test_data[['is_fraud', 'Predicted_Label', 'Predicted_Probability']].head())

# Save predictions to a CSV file
output_file_path = 'test_predictions.csv'
test_data[['is_fraud', 'Predicted_Label', 'Predicted_Probability']].to_csv(output_file_path, index=False)
print(f'Predictions saved to {output_file_path}')
