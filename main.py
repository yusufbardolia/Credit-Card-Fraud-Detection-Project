# fraud_detection.py

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import time

# Step 2: Load the Dataset
data = pd.read_csv('creditcard.csv')

# Step 3: Data Exploration (Optional)
print(data.head())
print(data.info())
print(data.describe())

# Step 4: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Feature Scaling
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Amount', 'Time'], axis=1)

# Handling Imbalanced Data with SMOTE
X = data.drop('Class', axis=1)
y = data['Class']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Corrected line to display value counts
print(y_res.value_counts())

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train and Evaluate Logistic Regression Model
print("Training Logistic Regression model...")
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train, y_train)
y_pred_log_reg = log_reg_model.predict(X_test)
print("Logistic Regression model training complete.")

print("Evaluating Logistic Regression model...")
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))
print(f'Logistic Regression ROC-AUC: {roc_auc_score(y_test, y_pred_log_reg)}')

# Plot ROC Curve for Logistic Regression
fpr, tpr, _ = roc_curve(y_test, y_pred_log_reg)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Logistic Regression ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# Train and Evaluate Random Forest Model
print("Starting Random Forest training...")
start_time = time.time()

# Reduced complexity model
rf_model = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42, n_jobs=-1)
print("Random Forest model initialized.")

# Start training
rf_model.fit(X_train, y_train)
print("Random Forest model training complete.")

end_time = time.time()
print(f"Random Forest model training complete. Time taken: {end_time - start_time} seconds")

# Evaluate model
print("Evaluating Random Forest model...")
y_pred_rf = rf_model.predict(X_test)
print("Random Forest evaluation complete.")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(f'Random Forest ROC-AUC: {roc_auc_score(y_test, y_pred_rf)}')

# Plot ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkgreen', lw=2, label='Random Forest ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Random Forest')
plt.legend(loc="lower right")
plt.show()

# Train and Evaluate Decision Tree Model
print("Training Decision Tree model...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree model training complete.")

print("Making predictions with Decision Tree...")
y_pred_dt = dt_model.predict(X_test)
print("Predictions complete.")

print("Evaluating Decision Tree model...")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print(f'Decision Tree ROC-AUC: {roc_auc_score(y_test, y_pred_dt)}')

# Plot ROC Curve for Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)
plt.figure()
plt.plot(fpr_dt, tpr_dt, color='darkblue', lw=2, label='Decision Tree ROC curve (area = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Decision Tree')
plt.legend(loc="lower right")
plt.show()

# Step 8: Save the Best Model
joblib.dump(rf_model, 'credit_fraud_detection_model.pkl')
