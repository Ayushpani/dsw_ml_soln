# Loan Default Prediction

## Project Overview
This project aims to build a robust classification model to predict loan default behavior for an NBFC (Non-Banking Financial Company). The goal is to identify potential defaulters based on historical data and improve the company's risk assessment and loan approval processes.

### Key Objectives
- **Data Analysis**: Perform exploratory data analysis (EDA) to understand the data and identify key predictors.
- **Feature Engineering**: Transform and preprocess features for optimal model performance.
- **Model Training**: Implement and compare multiple machine learning models, including Logistic Regression, Random Forest, and XGBoost.
- **Evaluation**: Use metrics like AUC, precision, recall, and F1-score to evaluate model performance.

---

## Project Structure

```
Loan Default Prediction
├── model_.py            # Script for training and saving models
├── model_selection.ipynb # Notebook for evaluating trained models
├── requirements.txt     # List of required libraries
├── data_prep.py         # Preprocessing and feature engineering script
├── X_features.csv       # Processed training feature data
├── y_labels.csv         # Processed training target labels
├── test_data.csv        # Test dataset
├── scaler.pkl           # Saved scaler object
├── logistic_regression_model.pkl # Saved Logistic Regression model
├── random_forest_model.pkl        # Saved Random Forest model
├── xgboost_model.pkl    # Saved XGBoost model
└── README.md            # Detailed project documentation
```

---

## How to Execute

### Prerequisites
Ensure you have Python 3.7+ installed. Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Steps to Run the Project
1. **Data Preparation**:
   - Ensure `X_features.csv` and `y_labels.csv` (processed training data) are present in the project directory.
   - Ensure `test_data.csv` (test dataset) is available for evaluation.

2. **Train the Models**:
   - Run the `model_.py` script to preprocess data, train models, and save them for evaluation.

```bash
python model_.py
```

3. **Evaluate Models**:
   - Use the `model_selection.ipynb` notebook to evaluate the trained models on test data.

4. **Check Results**:
   - Model performance metrics (AUC, precision, recall, etc.) will be displayed for each model.

---

## Methodology

### Exploratory Data Analysis (EDA)
1. **Overview of Data**:
   - Summary statistics, missing value analysis, and data types were reviewed.

2. **Correlation Analysis**:
   - Numeric features were analyzed for their correlation with the target variable (`loan_status`).
   - Categorical variables were encoded and analyzed for their predictive power.

3. **Outlier Detection**:
   - Boxplots and statistical methods (IQR) were used to identify and cap outliers.

4. **Time-Based Trends**:
   - `transaction_date` was transformed to extract `transaction_month` and `transaction_year` to analyze seasonal patterns.

#### EDA Flowchart:
1. Data Cleaning -> 2. Feature Analysis -> 3. Outlier Detection -> 4. Encoding Categorical Data -> 5. Correlation Analysis

### Feature Engineering
1. **Engineered Features**:
   - Created features like `interest_payment_ratio` to capture financial behavior.
   - Aligned test data features with training data for consistency.

2. **Scaling**:
   - Used `StandardScaler` to normalize numerical features for models sensitive to scale.

### Model Training and Evaluation
1. **Models Implemented**:
   - **Logistic Regression**: Baseline model for linear separability.
   - **Random Forest**: Robust against overfitting, captures feature importance.
   - **XGBoost**: Handles non-linear relationships efficiently.

2. **Evaluation Metrics**:
   - **AUC**: Measures the ability to rank positive and negative classes.
   - **Precision, Recall, F1-Score**: Detailed performance metrics for each class.

---

## Results
1. **Logistic Regression**:
   - AUC: 0.71
   - Strengths: Simple, interpretable baseline.

2. **Random Forest**:
   - AUC: 0.76
   - Strengths: Captures non-linear relationships and feature importance.

3. **XGBoost**:
   - AUC: 0.80
   - Strengths: Superior performance, handles missing data effectively.

---

## Conclusion
The XGBoost model outperformed Logistic Regression and Random Forest, achieving an AUC of 0.80. It is the recommended model for deployment to improve loan default prediction.

Future improvements could include:
- Hyperparameter tuning for further optimization.
- Incorporating additional features such as external financial indicators.

---

## Contact
For any questions or issues, feel free to reach out at [ayushpanigrahi84@gmail.com].
