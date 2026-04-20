import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def handle_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def main():
    try:
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'uber.csv'))
    except FileNotFoundError:
        print("Please download 'uber.csv' from https://www.kaggle.com/datasets/yasserh/uber-fares-dataset and place it in this folder.")
        return

    # 1. Pre-process the dataset
    print("Preprocessing data...")
    df = df.dropna()
    df = df.drop(['Unnamed: 0', 'key'], axis=1) # Drop identifiers
    
    # Extract datetime features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df = df.drop('pickup_datetime', axis=1)

    # 2. Identify outliers (and remove them for better modeling)
    print("Handling outliers...")
    numerical_cols = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    for col in numerical_cols:
        df = handle_outliers(df, col)

    # Filter invalid coordinates
    df = df[(df['pickup_latitude'] >= -90) & (df['pickup_latitude'] <= 90)]
    df = df[(df['dropoff_latitude'] >= -90) & (df['dropoff_latitude'] <= 90)]
    df = df[(df['pickup_longitude'] >= -180) & (df['pickup_longitude'] <= 180)]
    df = df[(df['dropoff_longitude'] >= -180) & (df['dropoff_longitude'] <= 180)]

    # 3. Check the correlation
    print("Checking correlation...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    print("Correlation matrix saved as 'correlation_matrix.png'")

    # 4. Implement regression models
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1)
    }

    # 5. Evaluate and compare models
    print("\nModel Evaluation:")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"--- {name} ---")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}\n")

if __name__ == "__main__":
    main()
