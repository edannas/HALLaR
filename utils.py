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

def generate_Y(n, r):
    """
    Generates a matrix Y of dimension n x r such that YY^T has trace 1.
    
    Parameters:
    n (int): Number of rows.
    r (int): Number of columns.
    
    Returns:
    Y (numpy.ndarray): Matrix of shape (n, r) with trace(YY^T) = 1.
    """

    Y = np.random.randn(n, r)

    # Initialize Y as a zero matrix
    # Y = np.zeros((n, r))
    
    # Set the first element to 1
    # Y[0, 0] = 1.0
    
    # Normalize Y to ensure the Frobenius norm is 1
    Y /= np.linalg.norm(Y, 'fro')

    # print(np.sum(np.square(Y)), np.linalg.norm(Y, 'fro'))
    
    return Y