import numpy as np
import matplotlib.pyplot as plt

def generate_Y(n, r):
    """
    Generates a matrix Y of dimension n x r such that YY^T has trace 1.
    
    Parameters:
    n (int): Number of rows.
    r (int): Number of columns.
    
    Returns:
    Y (numpy.ndarray): Matrix of shape (n, r) with trace(YY^T) = 1.
    """
    
    # Initialize Y as a random matrix
    Y = np.random.randn(n, r)
    
    # Normalize Y to ensure the Frobenius norm is 1
    Y /= np.linalg.norm(Y, 'fro')
    
    return Y

# Y_0 = generate_Y()
# numpy.savetxt("Y_0.csv", Y_0, delimiter=",")

def gradient_descent(L_beta, Y_0, epsilon, p, C, learning_rate=1e-3, max_iterations=1000):
    Y_k = Y_0
    iteration = 0

    while True:
        # computing gradient of L_beta at Y_k
        gradient = compute_gradient_fd(L_beta, Y_k, p)
        
        # updating Y_k using gradient descent
        Y_k -= learning_rate * gradient
        
        # checking convergence
        if np.linalg.norm(gradient) < epsilon or iteration >= max_iterations:
            #or np.linalg.norm(gradient-previous_gradient < 1e-5):
            break
        
        iteration += 1
        print(np.linalg.norm(gradient))
    
    return Y_k

def compute_gradient_fd(L_beta, Y): # computing gradient using finite differences
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

def plot_metrics(graph, beta, rank, objective_values, constraint_violations, reference_value):
    iterations = range(1, len(objective_values))
    fig, ax1 = plt.subplots()

    # Plot the objective function values
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Function Value', color='tab:blue')
    ax1.plot(iterations, objective_values, '-', color='tab:blue', label='Objective Function Value')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Add a horizontal reference line for the objective function value
    ax1.axhline(y=reference_value, color='tab:blue', linestyle='--', label='Reference Value (NetworkX)')

    # Create a second y-axis to plot the constraint violations
    ax2 = ax1.twinx()
    ax2.set_ylabel('Constraint Violation', color='tab:red')
    ax2.plot(iterations, constraint_violations, '-', color='tab:red', label='Constraint Violation')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels += labels2

    # Add the combined legend to the plot
    fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # plt.title(u'Lagrangian Value and Constraint Violation per Iteration ($β = {}$, r = {})'.format(beta, rank))
    plt.savefig("plots/{}_beta{}_rank{}_I{}.jpg".format(graph, beta, rank, iterations[-1]), dpi = 300)
    plt.show()