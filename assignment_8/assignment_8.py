import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    try:
        # Load a dummy/example cosmetics dataset if the exact one is not specified
        # Assume it's named 'cosmetics.csv' and has a target column 'Response'
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'cosmetics.csv'))
    except FileNotFoundError:
        print("Please place the cosmetics dataset named 'cosmetics.csv' in this folder.")
        print("Using dummy data for demonstration.")
        # Creating dummy data
        data = {
            'Age': [25, 30, 35, 40, 45, 50, 55, 60, 28, 32],
            'Income': [50000, 60000, 55000, 80000, 75000, 90000, 85000, 100000, 52000, 62000],
            'Previous_Purchases': [2, 5, 3, 8, 6, 10, 7, 12, 1, 4],
            'Response': [0, 1, 0, 1, 1, 1, 0, 1, 0, 1] # Target
        }
        df = pd.DataFrame(data)

    print("Data Preview:")
    print(df.head())

    X = df.drop('Sensitive', axis=1)
    y = df['Sensitive']

    # Encoding if necessary
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply Random Forest (or any appropriate classifier)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Results ---")
    print("Confusion Matrix:")
    print(cm)
    print(f"a. Accuracy:  {acc:.4f}")
    print(f"b. Precision: {prec:.4f}")
    print(f"c. Recall:    {rec:.4f}")
    print(f"d. F-1 score: {f1:.4f}")

if __name__ == "__main__":
    main()
