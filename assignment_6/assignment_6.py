import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from scipy.stats import skew, kurtosis
import statsmodels.api as sm

def perform_analysis(df, dataset_name):
    print(f"\n{'='*40}")
    print(f"Analysis for {dataset_name}")
    print(f"{'='*40}")

    # a. Univariate analysis
    print("\n--- a. Univariate Analysis ---")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"\nFeature: {col}")
            print(f"Mean: {df[col].mean():.4f}")
            print(f"Median: {df[col].median():.4f}")
            # Mode can have multiple values
            mode_val = df[col].mode()
            print(f"Mode: {mode_val[0] if not mode_val.empty else 'N/A'}")
            print(f"Variance: {df[col].var():.4f}")
            print(f"Standard Deviation: {df[col].std():.4f}")
            print(f"Skewness: {skew(df[col].dropna()):.4f}")
            print(f"Kurtosis: {kurtosis(df[col].dropna()):.4f}")

    # For Bivariate and Multiple Regression, let's assume 'Outcome' is the target for classification and 'Glucose' for regression
    target_cls = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]
    
    # b. Bivariate analysis: Linear and logistic regression modeling
    print("\n--- b. Bivariate Analysis ---")
    # Linear Regression (Predicting one feature using another, e.g., BMI from Glucose)
    if 'Glucose' in df.columns and 'BMI' in df.columns:
        X_bi_lin = df[['Glucose']].values
        y_bi_lin = df['BMI'].values
        valid_idx = ~np.isnan(X_bi_lin.flatten()) & ~np.isnan(y_bi_lin)
        if sum(valid_idx) > 0:
            lin_model = LinearRegression()
            lin_model.fit(X_bi_lin[valid_idx], y_bi_lin[valid_idx])
            print(f"Linear Regression (BMI ~ Glucose) R2: {lin_model.score(X_bi_lin[valid_idx], y_bi_lin[valid_idx]):.4f}")

    # Logistic Regression (Predicting Outcome using one feature, e.g., Glucose)
    if 'Glucose' in df.columns and target_cls in df.columns:
        X_bi_log = df[['Glucose']].values
        y_bi_log = df[target_cls].values
        valid_idx = ~np.isnan(X_bi_log.flatten()) & ~np.isnan(y_bi_log)
        if sum(valid_idx) > 0:
            log_model = LogisticRegression()
            log_model.fit(X_bi_log[valid_idx], y_bi_log[valid_idx])
            print(f"Logistic Regression (Outcome ~ Glucose) Accuracy: {log_model.score(X_bi_log[valid_idx], y_bi_log[valid_idx]):.4f}")

    # c. Multiple Regression analysis
    print("\n--- c. Multiple Regression Analysis ---")
    # Using statsmodels for summary
    X_mult = df.drop(target_cls, axis=1)
    y_mult = df[target_cls]
    
    # Handle NaNs for statsmodels
    temp_df = df.dropna()
    if not temp_df.empty:
        X_mult_sm = sm.add_constant(temp_df.drop(target_cls, axis=1))
        y_mult_sm = temp_df[target_cls]
        try:
            model = sm.OLS(y_mult_sm, X_mult_sm).fit()
            print("Multiple Linear Regression Summary (Predicting Outcome as continuous):")
            print(model.summary().tables[1]) # Print only coeff table for brevity
        except Exception as e:
            print("Could not fit Multiple Regression model:", e)

    # d. Comparison
    # This will be discussed in the final output
    print(f"\nFinished analysis for {dataset_name}.")
    return temp_df

def main():
    try:
        # Load datasets (Assuming both are downloaded, or we just process one if they are the same)
        # Note: Often "UCI diabetes" and "Pima Indians diabetes" essentially refer to the same dataset on Kaggle/UCI.
        print("Loading datasets...")
        # Since uci and pima might be the same, we'll try to load diabetes.csv
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'diabetes.csv'))
        perform_analysis(df, "Pima Indians Diabetes Dataset")
        
        # If there is a second dataset, load and analyze it here
        # df2 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'other_diabetes.csv'))
        # perform_analysis(df2, "Other Diabetes Dataset")
        
    except FileNotFoundError:
        print("Please download 'diabetes.csv' from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database and place it in this folder.")

if __name__ == "__main__":
    main()
