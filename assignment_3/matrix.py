import numpy as np
import cmath

def eigenvalues_numpy(matrix):
    """
    Find eigenvalues of a 2x2 matrix using NumPy.
    """
    matrix_np = np.array(matrix)
    # np.linalg.eig returns a tuple consisting of eigenvalues and eigenvectors
    eigenvalues, _ = np.linalg.eig(matrix_np)
    return eigenvalues

def eigenvalues_manual(matrix):
    """
    Find eigenvalues of a 2x2 matrix using the characteristic equation.
    For a 2x2 matrix [[a, b], [c, d]]:
    characteristic equation is: λ^2 - trace*λ + determinant = 0
    """
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    
    trace = a + d
    determinant = (a * d) - (b * c)
    
    # Calculate the discriminant (b^2 - 4ac in the quadratic formula)
    discriminant = (trace**2) - (4 * determinant)
    
    # Calculate the two eigenvalues (roots of the characteristic equation)
    # Using cmath to handle potential complex eigenvalues (negative discriminant)
    lambda_1 = (trace - cmath.sqrt(discriminant)) / 2
    lambda_2 = (trace + cmath.sqrt(discriminant)) / 2
    
    # If the imaginary part is zero, return only the real part for cleaner output
    if lambda_1.imag == 0 and lambda_2.imag == 0:
        return lambda_1.real, lambda_2.real
        
    return lambda_1, lambda_2

if __name__ == "__main__":
    # Define a 2x2 matrix
    # Example:
    # | 4  1 |
    # | 2  3 |
    A = [[4, 1], 
         [2, 3]]
    
    print("Matrix A:")
    for row in A:
        print(row)
        
    print("\n--- Using Characteristic Equation (Manual Formula) ---")
    val1, val2 = eigenvalues_manual(A)
    print(f"Eigenvalues: {val1}, {val2}")
    
    print("\n--- Using NumPy ---")
    try:
        np_vals = eigenvalues_numpy(A)
        print(f"Eigenvalues: {np_vals[0]}, {np_vals[1]}")
    except ImportError:
        print("NumPy is not installed. Run `pip install numpy` to use the NumPy method.")
