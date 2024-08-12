Credit Card Fraud Detection

Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions. By identifying fraudulent transactions, we can minimize financial losses and enhance transaction security. The project involves data preprocessing, model training, and evaluation using various machine learning algorithms.

Table of Contents
Project Description
Technologies Used
Dataset
Installation
Usage
Model Training
Model Testing
Evaluation
Contributing
License

Project Description

The project involves detecting fraudulent credit card transactions using a machine learning approach. The main tasks include:

Data Preprocessing: Handling missing values, feature scaling, and encoding categorical variables.
Model Training: Training various machine learning models including Logistic Regression, Random Forest, and Decision Tree.
Model Evaluation: Assessing model performance using metrics like confusion matrix, classification report, and ROC-AUC score.
Model Testing: Validating the model on a separate test dataset.

Technologies Used

Python: Programming language used for implementing the machine learning models and data preprocessing.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Scikit-learn: For machine learning algorithms and evaluation.
Imbalanced-learn: For handling class imbalance using SMOTE.
Matplotlib: For plotting evaluation metrics.
Joblib: For saving and loading the trained model.

Dataset

The dataset used for this project contains information about credit card transactions, including:

trans_date_trans_time: Transaction date and time.
cc_num: Credit card number.
merchant: Merchant name.
category: Transaction category.
amt: Transaction amount.
is_fraud: Target variable indicating whether the transaction is fraudulent.

Note: The dataset can be downloaded from Kaggle's Credit Card Fraud Detection Dataset.


Installation
To set up the project environment, follow these steps:

Clone the repository:


"git clone https:https://github.com/yusufbardolia/Credit-Card-Fraud-Detection-Project"

Navigate to the project directory:

"cd Credit-Card-Fraud-Detection-Project"

Create and activate a virtual environment:

"python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`"

Install the required packages:

pip install -r requirements.txt




Usage
To train the model, run the following script:

"python main.py"

This will:

Load and preprocess the dataset.
Train Logistic Regression, Random Forest, and Decision Tree models.
Evaluate the models and save the best model to credit_fraud_detection_model.pkl.
To test the model, ensure you have a test dataset in the same format and run:

python predict_fraud.py

This will:

Load the saved model.
Preprocess the test dataset.
Evaluate the model’s performance on the test dataset.

Model Training

The following machine learning algorithms were used:

Logistic Regression: A baseline model for binary classification.
Random Forest Classifier: An ensemble method that improves performance by combining multiple decision trees.
Decision Tree Classifier: Provides interpretability and decision-making insights.
Evaluation Metrics:

Confusion Matrix
Classification Report
ROC-AUC Score
ROC Curve
Model Testing
The model was tested on a separate dataset to validate its performance. The testing script preprocesses the test data in the same manner as the training data and evaluates the model’s performance.

Contributing
If you would like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch for your changes.
Commit your changes and push to the new branch.
Create a pull request with a description of your changes.

