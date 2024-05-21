import numpy as np

from main_hlr import A_adjoint, q

def gradient_descent(L_beta, Y_0, epsilon, p, C, learning_rate=1e-3, max_iterations=1000):
    Y_k = Y_0
    iteration = 0

    while True:
        # computing gradient of L_beta at Y_k
        gradient = compute_gradient(L_beta, Y_k, p)
        
        # updating Y_k using gradient descent
        Y_k -= learning_rate * gradient
        
        # checking convergence
        if np.linalg.norm(gradient) < epsilon or iteration >= max_iterations:
            #or np.linalg.norm(gradient-previous_gradient < 1e-5):
            break
        
        iteration += 1
        print(np.linalg.norm(gradient))
    
    return Y_k

def compute_gradient_fd(L_beta, Y):
    # computing gradient using finite differences
    eps = 1e-6
    grad = np.zeros_like(Y)
    n, r = Y.shape
    for i in range(n):
        for j in range(r):
            Y_plus_eps = Y.copy()
            Y_plus_eps[i, j] += eps
            L_beta_plus_eps = L_beta(np.dot(Y_plus_eps, Y_plus_eps.T))
            L_beta_Y = L_beta(np.dot(Y, Y.T))
            grad[i, j] = (L_beta_plus_eps - L_beta_Y) / eps
            
    return grad

def compute_gradient(L_beta, Y, p, C):
    # computing gradient using finite differences
    grad = C + A_adjoint(q(Y, p))
    return grad