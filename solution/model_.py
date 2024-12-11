import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle


class ClassificationModels:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {}

    def preprocess(self):
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def add_model(self, name, model):
        self.models[name] = model

    def train(self, model_name):
        model = self.models[model_name]
        model.fit(self.X_train, self.y_train)
        self.models[model_name] = model

    def evaluate(self, model_name):
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        print(f"{model_name} AUC: {auc_score:.4f}")
        print(classification_report(self.y_test, model.predict(self.X_test)))
        return auc_score

    def run_all(self):
        for name, model in self.models.items():
            print(f"Training and Evaluating {name}")
            self.train(name)
            self.evaluate(name)

# Example Usage
if __name__ == "__main__":
    # Load your prepared dataset
    X = pd.read_csv("X_features.csv")
    y = pd.read_csv("y_labels.csv")['loan_status']

    # Initialize and preprocess
    clf_models = ClassificationModels(X, y)
    clf_models.preprocess()

    # Add models
    clf_models.add_model("Logistic Regression", LogisticRegression(max_iter=1000))
    clf_models.add_model("Random Forest", RandomForestClassifier(random_state=42))
    clf_models.add_model("Gradient Boosting", GradientBoostingClassifier(random_state=42))

    # Run all models
    clf_models.run_all()

    from sklearn.preprocessing import StandardScaler

    # Save the models with pickle
    for name, model in clf_models.models.items():
        with open(f"{name.replace(' ', '_').lower()}_model.pkl", "wb") as f:
            pickle.dump(model, f)

    # Save the scaler with pickle
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Models and scaler saved successfully!")
