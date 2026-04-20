import pandas as pd
import numpy as np

# Step 1: Import required libraries
# (Included above)

# Step 2: Create a simple dataset
data = {
    'Math': [85, 90, 78, 92, 88],
    'Science': [80, 85, 75, 95, 90]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Step 3: Standardize the data (Mean = 0, Std = 1)
df_standardized = (df - df.mean()) / df.std()

print("\nStandardized Data:")
print(df_standardized)

# Step 4: Compute Covariance Matrix
cov_matrix = df_standardized.cov()

print("\nCovariance Matrix:")
print(cov_matrix)

# Step 5: Compute Eigenvalues and Eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nEigen Values:")
print(eigen_values)

print("\nEigen Vectors:")
print(eigen_vectors)

# Step 6: Select the principal component (largest eigenvalue)
principal_component = eigen_vectors[:, 0]

print("\nPrincipal Component:")
print(principal_component)

# Step 7: Transform the data (Project onto principal component)
projected_data = df_standardized.dot(principal_component)

print("\nProjected Data (1D PCA Result):")
print(projected_data)