import pandas as pd
import os

def main():
    filename = 'data.csv'
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found. Please ensure the file is in the correct directory.")
        return
    
    # Read the data
    print(f"Reading {filename}...")
    df = pd.read_csv(filename)
    
    # Sort by 'marks1' in ascending order
    # You can change 'marks1' to 'marks2' if needed, or sort by both: ['marks1', 'marks2']
    print("\n--- Sorting by marks1 (Ascending) ---")
    sorted_df = df.sort_values(by='marks1', ascending=True)
    
    # Display the result
    print(sorted_df)
    
    # Optional: Save to a new file
    # sorted_df.to_csv('sorted_data.csv', index=False)
    # print("\nSorted data saved to 'sorted_data.csv'")

if __name__ == "__main__":
    main()
