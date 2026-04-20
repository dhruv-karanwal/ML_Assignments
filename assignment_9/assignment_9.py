import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def main():
    try:
        # Load dataset
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Mall_Customers.csv'))
    except FileNotFoundError:
        print("Please download 'Mall_Customers.csv' and place it in this folder.")
        # Alternatively, using the one present in archive directory
        try:
            df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive', 'Mall_Customers.csv'))
            print("Loaded from archive folder.")
        except FileNotFoundError:
            return

    # a. Apply Data pre-processing
    print("Preprocessing data...")
    # Drop CustomerID as it's not useful for clustering
    df = df.drop('CustomerID', axis=1)

    # Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # b. Perform data-preparation (Train-Test Split)
    # Note: Train-Test Split is typically for supervised learning. 
    # For clustering, we usually fit on the whole dataset or a training set.
    # The prompt asks for it, so we do it.
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # We will cluster based on Annual Income and Spending Score as per standard practice, or all features.
    # The prompt says "based on Spending Score". We'll use Income and Spending Score.
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X_train_scaled = StandardScaler().fit_transform(X_train[features])
    X_test_scaled = StandardScaler().fit_transform(X_test[features])
    X_full_scaled = StandardScaler().fit_transform(df[features])

    # c. Apply Machine Learning Algorithm (Clustering)
    print("Applying Clustering Algorithms...")
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_full_scaled)

    # 2. Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=5)
    agglo_labels = agglo.fit_predict(X_full_scaled)

    # d. Evaluate Model
    print("\n--- Model Evaluation (Silhouette Score) ---")
    kmeans_score = silhouette_score(X_full_scaled, kmeans_labels)
    agglo_score = silhouette_score(X_full_scaled, agglo_labels)
    
    print(f"K-Means Silhouette Score: {kmeans_score:.4f}")
    print(f"Agglomerative Clustering Silhouette Score: {agglo_score:.4f}")

    # Visualize K-Means clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=kmeans_labels, cmap='viridis')
    plt.title('K-Means Clustering of Mall Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.savefig('kmeans_clusters.png')
    print("Cluster visualization saved as 'kmeans_clusters.png'")

if __name__ == "__main__":
    main()
