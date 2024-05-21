import numpy as np

def initialize_U(n, r):
    # Create a zero matrix of shape n x r
    U_0 = np.zeros((n, r))

    # Fill the diagonal of the top-left square block with ones
    min_dim = min(n, r)
    U_0[:min_dim, :min_dim] = np.eye(min_dim)

    # Normalize U_0 to have trace 1
    U_0 = U_0 / np.sqrt(np.trace(np.dot(U_0, U_0.T)))

    return U_0

def calculate_Y(X, s):
    # calculating U ∈ (n x s) where UUT a s-rank approximation of X
    # Perform eigen decomposition on X
    eigvals, eigvecs = np.linalg.eigh(X)
    
    # Take the square root of the largest s eigenvalues
    largest_eigvals = eigvals[-s:]
    Y = np.sqrt(largest_eigvals) * eigvecs[:, -s:]
    
    return Y

def initialize_X(n):
    # Generate a random symmetric matrix
    X = np.random.randn(n, n)
    X = (X + X.T) / 2  # Ensure symmetry

    # Make it positive semidefinite by adding a small multiple of the identity matrix
    eps = 1e-6
    X += eps * np.eye(n)

    # Normalize the matrix to ensure trace = 1
    trace_X = np.trace(X)
    X /= trace_X

    return X

def compute_gradient(L_beta, Y, p, C):
    # computing gradient using finite differences
    grad = C + A_adjoint(q(Y, p))
    return grad