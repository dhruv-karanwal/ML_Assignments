import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    try:
        # Load dataset
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Admission_Predict.csv'))
    except FileNotFoundError:
        print("Please download 'Admission_Predict.csv' from https://www.kaggle.com/mohansacharya/graduate-admissions and place it in this folder.")
        return

    # Data transformation
    # The dataset provides a "Chance of Admit" as a continuous value (0 to 1). 
    # To convert it to a classification problem target (Admitted: 0 or 1), we apply a threshold.
    # Let's say chance >= 0.75 means admitted (1), else not admitted (0).
    target_col = 'Chance of Admit '
    if target_col in df.columns:
        df['Admitted'] = (df[target_col] >= 0.75).astype(int)
        df = df.drop(target_col, axis=1)
    
    if 'Serial No.' in df.columns:
        df = df.drop('Serial No.', axis=1)

    X = df.drop('Admitted', axis=1)
    y = df['Admitted']

    # a. Perform data-preparation (Train-Test Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # b. Apply Machine Learning Algorithm (Decision Tree)
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    # c. Evaluate Model
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
