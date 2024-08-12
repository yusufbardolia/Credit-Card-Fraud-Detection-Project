# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Step 2: Load the Dataset
data = pd.read_csv(r'c:\Users\yusuf\Downloads\archive (4)\fraudTrain.csv')

# Step 3: Data Preprocessing
# Print column names to verify them
print("Column names in the dataset:", data.columns)

# Convert categorical columns to numeric
label_encoders = {}
categorical_columns = ['merchant', 'category', 'gender', 'job']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Convert 'trans_date_trans_time' to UNIX timestamp
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['trans_date_trans_time'] = data['trans_date_trans_time'].astype(np.int64)  # Use UNIX timestamps

# Feature Scaling
scaler = StandardScaler()
data['scaled_amt'] = scaler.fit_transform(data[['amt']])
data['scaled_trans_time'] = scaler.fit_transform(data[['trans_date_trans_time']])

# Drop original and unnecessary columns
data = data.drop(['amt', 'trans_date_trans_time', 'Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long'], axis=1)

# Handling Imbalanced Data with SMOTE
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Print value counts of the resampled data
print(y_res.value_counts())

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train and Evaluate Logistic Regression Model
log_reg_model = LogisticRegression(random_state=42, C=0.1)
log_reg_model.fit(X_train, y_train)
y_pred_log_reg = log_reg_model.predict(X_test)
print("Logistic Regression model training complete.")
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
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    max_features='sqrt',
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest model training complete.")
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
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree model training complete.")
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
