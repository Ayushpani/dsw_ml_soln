# data_prep.py
import pandas as pd

def preprocess_test_data(test_data):
    # Drop unwanted columns
    features_to_drop = ['transaction_month', 'month', 'transaction_year', 'year', 'sub_grade_encoded', 'total_no_of_acc', 'C5']
    test_data = test_data.drop(columns=features_to_drop, errors='ignore')
    
    # Handle missing values if any
    test_data.fillna(0, inplace=True)
    
    # Create any engineered features
    test_data['interest_payment_ratio'] = test_data['installment'] / test_data['annual_inc']

    # Ensure test data has the same feature order and columns as training
    test_data = test_data.select_dtypes(include=['number'])  # Keep only numeric columns
    return test_data
