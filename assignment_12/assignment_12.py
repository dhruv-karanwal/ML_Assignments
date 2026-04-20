import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def main():
    try:
        # Load dataset
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    except FileNotFoundError:
        print("Please download 'Telco Customer Churn' dataset from https://www.kaggle.com/datasets/blastchar/telco-customer-churn and place it herein.")
        return

    # a. Preprocess data
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Handle TotalCharges missing values (blank strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # b. Train Random Forest classifier and tune hyperparameters
    print("Training Random Forest with Hyperparameter Tuning...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_

    # c. Evaluate model
    y_pred = best_rf.predict(X_test_scaled)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # d. Analyze feature importance
    feature_importances = best_rf.feature_importances_
    features = X.columns
    indices = np.argsort(feature_importances)[::-1]

    print("\n--- Feature Importance ---")
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {features[indices[f]]} ({feature_importances[indices[f]]:.4f})")

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")

if __name__ == "__main__":
    main()
