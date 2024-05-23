import numpy as np

def initialize_X(n):
    # Generate a random symmetric matrix
    X = np.random.randn(n, n)
    X = (X + X.T) / 2

    # Make it positive semidefinite by adding a small multiple of the identity matrix
    eps = 1e-6
    X += eps * np.eye(n)

    # Normalize the matrix to ensure trace = 1
    trace_X = np.trace(X)
    X /= trace_X

    return X

def calculate_Y(X, s): # calculating Y ∈ (n x s) where YY^T a s-rank approximation of X
    # Perform eigen decomposition on X
    eigvals, eigvecs = np.linalg.eigh(X)
    
    # Take the square root of the largest s eigenvalues 
    # (Y is the matrix of the values multiplied with corresponding eigenvectors)
    largest_eigvals = eigvals[-s:]
    Y = np.sqrt(largest_eigvals) * eigvecs[:, -s:]
    
    return Y