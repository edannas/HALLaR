# g instead of L_beta
# ambiguous theta
# check stopping criteria
# fix input parameters (inside ball)

from torch.autograd.functional import jacobian
from torch import tensor
import numpy as np
import random

def theta(Y, g):
    # Defining input tensor
    x = tensor(Y.dot(Y.T), requires_grad=True)

    # Compute the Jacobian using PyTorch
    jacobian_tensor = jacobian(g, x)

    # Converting the Jacobian tensor to a NumPy array
    jacobian_array = jacobian_tensor.detach().numpy()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(jacobian_array)

    # Find the minimum eigenvalue
    min_eigenvalue_index = np.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_eigenvalue_index]

    # Corresponding eigenvector
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]

    # Return the minimum eigenvalue and its corresponding eigenvector
    return jacobian_array, max(-min_eigenvalue, 0), min_eigenvector

def compute_alpha_k(Y_k, y_k, g):
    """
    Compute the value of alpha_k that minimizes the objective function.

    Parameters:
        Y_k (array_like): The matrix Y_k.
        y_k (array_like): The vector y_k.

    Returns:
        float: The value of alpha_k.
    """
    m = Y_k.shape[0]
    n = Y_k.shape[1]

    min_loss = float('inf')
    best_alpha = 0

    for alpha in np.linspace(0, 1, num=100):  # Adjust num for finer resolution
        # Compute the convex combination matrix
        convex_combination = g(alpha * np.outer(y_k, y_k) + (1 - alpha) * np.dot(Y_k, Y_k.T))

        # Compute the loss (Frobenius norm of the difference)
        loss = np.linalg.norm(convex_combination - np.dot(Y_k, Y_k.T), ord='fro')

        # Update minimum loss and best alpha if applicable
        if loss < min_loss:
            min_loss = loss
            best_alpha = alpha

    return best_alpha

# fix below function
def numerical_gradient(X_star, phi_s, epsilon = 1e-6):
    n, m = X_star.shape
    gradient = np.zeros_like(X_star)
    
    for i in range(n):
        for j in range(m):
            # Perturb the element at position (i, j)
            X_perturbed = X_star.copy()
            X_perturbed[i, j] += epsilon
            
            # Compute the numerical approximation of the partial derivative
            gradient[i, j] = (phi_s(X_perturbed) - phi_s(X_star)) / epsilon
    
    return gradient

def affine_approximation(function, x_bar, x):
        # Compute the gradient of function at point
        grad = numerical_gradient(x_bar, function)
    
        # Compute the affine approximation
        affine_approx = function(x_bar) + np.dot(grad.flatten(), (x - x_bar).flatten())
    
        return affine_approx

def adap_fista(x0, mu, L, phi_s, phi_n, sigma = 1, chi = .5):
    y_i = x_i = x0
    A_i = 0
    tau_i = 1
    i = 0 
    L_i = L

    while True:
        a_i = (tau_i + np.sqrt(tau_i**2 + 4 * tau_i * A_i(L_i - mu))) / (2 * (L_i - mu))
        x_i_tilde = (A_i * y_i + a_i * x_i) / (A_i * a_i)

        def affine_approx_phi_s(x):
            # Compute the gradient of function at point
            grad = numerical_gradient(x_i_tilde, 1e-6, phi_s)
    
            # Compute the affine approximation
            affine_approx = function(x_i_tilde) + np.dot(grad.flatten(), (x - x_i_tilde).flatten())
    
            return affine_approx
        
        def q_i(u):
            return affine_approx_phi_s(u) + phi_n(u) + L_i/2 * np.linalg.norm(u - x_i_tilde)**2
        
        initial_guess = np.zeros((n, m))  # Replace with your initial guess
    
        # Bounds for the matrix elements (optional)
        bounds = [(-np.inf, np.inf) for _ in range(n * m)]  # Replace with your bounds
    
        # Minimize the function q_i(u)
        result = minimize(q_i, initial_guess.flatten(), bounds=bounds)  # Minimize the flattened u
    
        # Reshape the optimized u to the original shape
        u_optimal = result.x.reshape((n, m))


def delta_B1(X):
    if np.linalg.norm(X, ord='fro') <= 1: # is the boundary included in the ball?
        return 0
    else: 
        return np.inf

def adap_aipp(g, lambda_0, rho, W):
    W_j_1 = W
    j = 1
    lambda_j = lambda_0
    M_j = 1
    M_j = random.randint(1, M_j)

    while j < 10:
        x_0 = W_j_1
        mu = 1/2
        L = M_j
        # ---> check below computation
        def phi_s(W): # check input variable
            return lambda_j * g(W) + 1/2 * np.linalg.norm(- W_j_1, ord='fro') ** 2
        def phi_n(W): # check input variable
            lambda_j * delta_B1(W)
        
        try:
            # ---> check adap_fista call
            W,V,L = adap_fista(x_0, mu, L, phi_s, phi_n, sigma = 1, chi = 1)

        except:
            lambda_j = lambda_j/2
            continue
        # ---> check condition <---
        condition = lambda_j * np.linalg.norm(W_j_1) - (lambda_j * g(W) + 1/2 * np.linalg.norm(W - W_j_1, ord='fro')**2) >= np.trace(np.dot(V, W_j_1 - W))
        if not condition:
            lambda_j = lambda_j/2
            continue
    
        lambda_j, M_j = lambda_j, L
        W_j, V_j = W, V
        R_j = (V_j + W_j_1 - W_j) / lambda_j

        if np.linalg.norm(R_j, ord='fro') <= rho:
            return W_j, R_j
        
        W_j_1 = W_j
        j = j + 1

    return W_j, R_j

def fw():
    pass

def hlr(Y_0, g, epsilon, rho, lambda_0):
    Y_k = Y_0
    s = 1
    k = 1

    while k < 10:
        Y_k = adap_aipp(g, lambda_0, rho, Y_k)

        G_k, theta_k, v_min = theta(Y_k, g)

        y_k = 0

        if theta_k > 0:
            y_k = v_min
    
        epsilon_k = np.trace(np.dot(G_k, Y_k)) + theta_k

        if epsilon_k < epsilon:
            return Y_k, theta_k

        # Compute alpha_k
        alpha_k = compute_alpha_k(Y_k, y_k, g)
        print("alpha_k:", alpha_k)

        if int(alpha_k) == 1: 
            Y_k, s = y_k, 1
    
        else:
            Y_k, s = [np.sqrt(1-alpha_k)*Y_k, np.sqrt(alpha_k)*y_k], s+1

        k = k + 1