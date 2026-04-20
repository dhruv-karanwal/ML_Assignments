import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'temperatures.csv'))
    except FileNotFoundError:
        print("Please download 'temperatures.csv' from https://www.kaggle.com/venky73/temperatures-of-india and place it in this folder.")
        return

    print("Data Preview:")
    print(df.head())

    # Assuming the dataset has columns 'YEAR' and month columns (JAN, FEB, etc.) and 'ANNUAL'
    # We want to predict Month-wise temperature. Let's predict a specific month based on YEAR, 
    # or reshape the data to predict Temperature based on YEAR and MONTH.
    
    # We'll build a simple regression model to predict the Annual temperature over the Years 
    # to visualize it easily as requested "Visualize a simple regression model."
    if 'YEAR' in df.columns and 'ANNUAL' in df.columns:
        X = df[['YEAR']]
        y = df['ANNUAL']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # a. Apply Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # b. Assess the performance
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n--- Model Evaluation (Predicting ANNUAL temp based on YEAR) ---")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R-Square: {r2:.4f}")

        # c. Visualize a simple regression model
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Actual Data')
        plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
        plt.title('Simple Linear Regression: Annual Average Temperature in India')
        plt.xlabel('Year')
        plt.ylabel('Annual Temperature (Celsius)')
        plt.legend()
        plt.savefig('temperature_regression.png')
        print("Visualization saved as 'temperature_regression.png'")

if __name__ == "__main__":
    main()
