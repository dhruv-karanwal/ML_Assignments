import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set plot style
sns.set(style="whitegrid")

def main():
    filename = 'data.csv'
    
    try:
        df = pd.read_csv(filename)
        print("Data Loaded Successfully!")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    # ==========================================
    # a. Compute and display summary statistics
    # ==========================================
    print("\n" + "="*50)
    print("Part A: Summary Statistics")
    print("="*50)
    
    # We'll focus on numerical columns for detailed stats
    numerical_cols = ['Age', 'marks1', 'marks2']
    print(f"Numerical Columns: {numerical_cols}")
    
    # Using describe() for most stats (count, mean, std, min, 25%, 50%, 75%, max)
    print("\n--- Descriptive Statistics ---")
    print(df[numerical_cols].describe().T)
    
    # Calculating specific stats manually
    print("\n--- Additional Statistics ---")
    for col in numerical_cols:
        if col in df.columns:
            print(f"\nStatistics for {col}:")
            print(f"  Range: {df[col].max() - df[col].min()}")
            print(f"  Variance: {df[col].var()}")
            print(f"  Skewness: {df[col].skew()}")

    # ==========================================
    # b. Illustrate feature distributions using histogram
    # ==========================================
    print("\n" + "="*50)
    print("Part B: Feature Distributions (Histograms)")
    print("="*50)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['Age'].dropna(), kde=True, bins=10, color='skyblue')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['marks1'].dropna(), kde=True, bins=10, color='salmon')
    plt.title('Distribution of Marks1')
    plt.xlabel('Marks1')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    print("Saving histograms to 'distribution_plots.png'...")
    plt.savefig('distribution_plots.png')
    # plt.show() 

   
    print("\n" + "="*50)
    print("Part C: Data Cleaning, Transformation & Model Building")
    print("="*50)
    
    print("1. Data Cleaning")
    print("Missing values before cleaning:\n", df.isnull().sum())
    
    for col in ['Age', 'marks1', 'marks2']:
        df[col] = df[col].fillna(df[col].mean())
        
    if df['result'].isnull().sum() > 0:
        print(f"Dropping {df['result'].isnull().sum()} rows with missing target.")
        df = df.dropna(subset=['result'])
        
    print("Missing values after cleaning:\n", df.isnull().sum())

    print("\n2. Data Transformation")
    if 'Gender' in df.columns:
        df['Gender_encoded'] = df['Gender'].map({'male': 0, 'female': 1})
        print("Encoded 'Gender': male=0, female=1")
    
    features = ['Age', 'marks1', 'marks2']
    if 'Gender_encoded' in df.columns:
        features.append('Gender_encoded')
    
    X = df[features]
    y = df['result']
    
    print("Features selected:", features)
    
    # --- 3. Model Building (Classification) ---
    print("\n3. Model Building (Logistic Regression)")
    
    # Split data
    # Ensure usage of stratify if possible, but with small data it might crash if classes are too few.
    # Given the small size (20 rows), just random split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0)) 
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
