import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

def main():
    try:
        # Load dataset
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Mall_Customers.csv'))
    except FileNotFoundError:
        print("Please download 'Mall_Customers.csv' and place it in this folder.")
        try:
            df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive', 'Mall_Customers.csv'))
            print("Loaded from archive folder.")
        except FileNotFoundError:
            return

    # a. Preprocess data (missing values, encoding, scaling)
    print("Preprocessing data...")
    # Grouping customers based on Annual Income and Spending Score
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

    # Scaling is optional depending on the range differences, but good practice
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # b. Apply Hierarchical Clustering and plot dendrogram
    print("Plotting Dendrogram...")
    plt.figure(figsize=(10, 7))
    plt.title("Customer Dendrogram")
    dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.savefig('dendrogram.png')
    print("Dendrogram saved as 'dendrogram.png'")

    # Apply Agglomerative Clustering
    # Based on standard Mall Customers analysis, 5 clusters is typically optimal
    optimal_clusters = 5
    print(f"\nApplying Agglomerative Clustering with {optimal_clusters} clusters...")
    hc = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X_scaled)

    # Add the cluster labels to the original dataframe
    df['Cluster'] = y_hc

    # c. Visualize clusters (Income vs Spending Score)
    plt.figure(figsize=(10, 7))
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    for val in range(optimal_clusters):
        plt.scatter(X[y_hc == val, 0], X[y_hc == val, 1], s=100, c=colors[val], label=f'Cluster {val}')
    
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.savefig('clusters.png')
    print("Cluster visualization saved as 'clusters.png'")

    # d. Interpret and describe each customer segment
    print("\n--- Cluster Interpretation ---")
    print("Look at the plotted 'clusters.png' to describe segments.")
    print("Typical segmentation for 5 clusters:")
    print("Cluster 0: Moderate Income, Moderate Spending (Average)")
    print("Cluster 1: High Income, Low Spending (Careful)")
    print("Cluster 2: Low Income, Low Spending (Sensible)")
    print("Cluster 3: Low Income, High Spending (Careless)")
    print("Cluster 4: High Income, High Spending (Target/Profitable)")

if __name__ == "__main__":
    main()
