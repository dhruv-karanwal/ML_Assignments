import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    try:
        print("Loading dataset...")
        # Note: Usually named UCI_Credit_Card.csv
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'UCI_Credit_Card.csv'))
    except FileNotFoundError:
        print("Please download 'UCI_Credit_Card.csv' from https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset and place it in this folder.")
        return

    # a. Preprocess data (cleaning, encoding, scaling, train-test split)
    print("Preprocessing data...")
    # Drop ID column
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    # Rename target column for convenience
    target_col = 'default.payment.next.month'
    
    # Handle missing values if any (dataset is usually clean)
    df = df.dropna()

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # b. Build SVM models
    print("Building SVM models...")
    svm_kernels = ['linear', 'poly', 'rbf']
    for kernel in svm_kernels:
        print(f"Training SVM ({kernel})...")
        model = SVC(kernel=kernel, max_iter=1000) # limiting iter for time in large dataset
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[f'SVM - {kernel}'] = {'accuracy': acc, 'confusion_matrix': cm}

    # c. Build KNN models
    print("Building KNN models...")
    k_values = [3, 5, 7]
    for k in k_values:
        print(f"Training KNN (k={k})...")
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[f'KNN - k={k}'] = {'accuracy': acc, 'confusion_matrix': cm}

    # e. Compare model accuracy and confusion matrices
    print("\n--- Model Comparison ---")
    for name, metrics in results.items():
        print(f"\nModel: {name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

if __name__ == "__main__":
    main()
