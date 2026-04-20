import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    try:
        # Load dataset
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Iris.csv'))
    except FileNotFoundError:
        print("Please download 'Iris.csv' from https://www.kaggle.com/datasets/uciml/iris and place it in this folder.")
        return

    # Preprocess
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # We cluster based on the features, not the target 'Species'
    X = df.drop('Species', axis=1) if 'Species' in df.columns else df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine the number of clusters using the elbow method
    wcss = [] # Within-Cluster-Sum-of-Squares
    max_k = 10
    
    print("Running k-means for different K values to compute WCSS...")
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('The Elbow Method showing the optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.savefig('elbow_method.png')
    print("Elbow method plot saved as 'elbow_method.png'")
    
    # Apply K-Means with the optimal number of clusters (which is 3 for Iris dataset)
    optimal_k = 3
    print(f"\nApplying K-Means with optimal clusters: {optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the dataframe
    df['Cluster'] = y_kmeans
    
    print("\nFirst 10 rows with Cluster column:")
    print(df.head(10))

if __name__ == "__main__":
    main()
