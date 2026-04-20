
import pandas as pd
import numpy as np
import os


def main():
    filename = 'data.csv'
    
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found. Please ensure the file is in the correct directory.")
        return
    
    print("\n--- a. Reading Data ---")
    df = pd.read_csv(filename)
    print("Data Read Successfully:\n", df.head())


    print("\n--- b. Shape of Data ---")
    print("Shape (Rows, Columns):", df.shape)


    print("\n--- c. Missing Values ---")
    print(df.isnull().sum())


    print("\n--- d. Data Types ---")
    print(df.dtypes)


    print("\n--- e. Finding Zeros ---")

    print((df == 0).sum())


    print("\n--- f. Indexing, Selecting, and Sorting ---")

    print("\nSelected Columns (Name, Age, marks1):")
    print(df[['Name', 'Age', 'marks1']].head())
    

    print("\nRows where Age > 30:")
    print(df[df['Age'] > 30].head())


    print("\nSorted by marks1 (Descending):")
    print(df.sort_values(by='marks1', ascending=False).head())


    print("\n--- g. Describe Attributes ---")
    print(df.describe(include='all'))


    print("\n--- h. Misc Operations ---")
    print("Unique Gender values:", df['Gender'].unique())
    print("Count of Unique Gender:\n", df['Gender'].value_counts())
    
    print("\nConverting 'marks1' from float to Int64")

    df['marks1_Int'] = df['marks1'].fillna(0).astype(int)
    print("New Column 'marks1_Int' type:", df['marks1_Int'].dtype)
    print(df[['marks1', 'marks1_Int']].head())
    

if __name__ == "__main__":
    main()
