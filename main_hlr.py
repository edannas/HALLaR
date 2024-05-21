"""
QUESTIONS:
store all iterations? (p-vector, U-vector, etc.)

min {C•UUT : A(UUT)=b, ||U||≤1, U ∈ R^{n×r}}

max {ee^T•X : X_ij=0, ij ∈ E, tr(X)=1, X ∈ S^n}

LANCElOT?

Check satisfying constraints and if approx optimal for the SDP, using duality gap?

On a really small graph maybe you know what is the optimal stable set? 

We dont necesarrily need THE optimal solution because of relaxation, but if almost it would be good. 

15-20 for report. 

Compute gradient using adjoint instead of finite difference, but check correctness with FD

"""

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import networkx as nx
from functions import initialize_U, opt_solver #evaluate_q_A,
from hlr import hlr
from scipy.optimize import minimize
#from gradient import gradient_descent

# --------------------------------------------------------------
# Defining SDP for Maximum stable set
# Given a graph G = ([n], E)

def import_graph_from_mtx(file_path):
    # reading graph adjacency matrix from .mtx file
    adjacency_matrix = mmread(file_path)
    
    # converting to NetworkX graph
    #graph = nx.to_scipy_sparse_array(adjacency_matrix)
    graph = nx.convert_matrix.from_scipy_sparse_array(adjacency_matrix)
    # Get the node set [n] and edge set E
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    
    return nodes, edges

file_path = "graphs/small.mtx"
nodes, edges = import_graph_from_mtx(file_path)

import networkx as nx
def create_small_graph():
    # Define nodes
    nodes = [1, 2, 3, 4, 5]
    # Define edges (as tuples of node pairs)
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    return nodes, edges

nodes, edges = create_small_graph()
# max {ee^T•X : X_ij=0, ij ∈ E, tr(X)=1, X ∈ S^n}

n = len(nodes)
m = len(edges)
e = np.ones(n) # m / n??
C = -np.outer(e, e)
b = np.zeros(m)

def A(X):
    AX = np.empty(m)
    for k, (i, j) in enumerate(edges): 
        # Assuming X is a sparse matrix representation of the adjacency matrix
        # print(AX[k], X[nodes.index(i), nodes.index(j)])
        AX[k] = X[nodes.index(i), nodes.index(j)]
    return AX

import cvxpy as cp
"""
def A_cp(X):
    # Define the optimization variable
    AX = cp.Variable(m)

    # Define the constraints (optional, depending on your problem)
    # constraints = []

    # Iterate over the edges
    for k, (i, j) in enumerate(edges): 
        # Assuming X is a sparse matrix representation of the adjacency matrix
        AX[k] = X[nodes.index(i), nodes.index(j)]

    return AX
"""

def A_cp(X):
    # Define the optimization variable
    m = len(edges)
    AX = cp.Variable(m)

    # Iterate over the edges and construct the constraints
    constraints = []
    for k, (i, j) in enumerate(edges):
        constraints.append(AX[k] == X[nodes.index(i), nodes.index(j)])

    # Define the problem
    problem = cp.Problem(cp.Minimize(0), constraints)

    # Solve the problem (this is just to enforce the constraints)
    problem.solve()

    return AX.value

def A_adjoint(v):
    M = np.zeros((n, n))
    for k, (i, j) in enumerate(edges):
        M[nodes.index(i), nodes.index(j)] = v[k]
    return M

def q(Y, p, beta):
    return p + beta * (A(Y.dot(Y.T)) - b)

def theta_tilde(Y, p, beta):
    # computing the minimum eigenvalue of C + A(q(Y; p))
    min_eigenvalue = np.linalg.eigvalsh(C + A_adjoint(q(Y, p, beta))).min()
    
    return max(-min_eigenvalue, 0)

def gradient_descent(L_beta, Y_0, epsilon, p, beta, learning_rate=1e-3, max_iterations=1000):
    Y_k = Y_0
    iteration = 0

    while True:
        # computing gradient of L_beta at Y_k
        gradient = compute_gradient2(L_beta, Y_k, p, beta)
        
        # updating Y_k using gradient descent
        Y_k -= learning_rate * gradient
        
        # checking convergence
        if np.linalg.norm(gradient) < epsilon or iteration >= max_iterations:
            #or np.linalg.norm(gradient-previous_gradient < 1e-5):
            break
        
        iteration += 1
        print(np.linalg.norm(gradient))
    
    return Y_k

def compute_gradient2(L_beta, Y, p, beta):
    # computing gradient using finite differences
    grad = C + A_adjoint(q(Y, p, beta))
    return grad
"""
def constraint_function(YY_flat):
    # Reshape YY_flat into a symmetric matrix YY^T
    n = int(np.sqrt(len(YY_flat)))
    YY = csr_matrix(YY_flat.reshape((n, n)))

    # Define the constraints
    constraints = []
    constraints.append(np.trace(YY.toarray()) - 1)  # trace(YY^T) ≤ 1

    # Check if YY^T is positive semidefinite
    eigenvalues = np.linalg.eigvalsh(YY.toarray())
    for eigval in eigenvalues:
        constraints.append(eigval >= 0)

    return constraints
"""
"""
def constraint_function(YY_flat):
    n = int(np.sqrt(len(YY_flat)))
    YY = csr_matrix(YY_flat.reshape((n, n)))

    # Define the constraints
    constraints = []
    constraints.append(YY.trace() - 1)  # trace(YY^T) ≤ 1

    # Check if YY^T is positive semidefinite
    eigenvalues = np.linalg.eigvalsh(YY.toarray())
    for eigval in eigenvalues:
        constraints.append(eigval >= 0)

    return constraints
"""
def constraint_function(YY_flat):
    n = int(np.sqrt(len(YY_flat)))
    YY = np.reshape(YY_flat, (n, n))

    # symmetric?

    # Initialize constraints
    constraints = []

    # Add trace constraint
    trace_constraint = np.trace(YY) - 1
    constraints.append(trace_constraint)

    # Check positive semidefiniteness
    eigenvalues = np.linalg.eigvalsh(YY)
    for eigval in eigenvalues:
        constraints.append(eigval)

    return constraints

def check_constraints_U(U_0, epsilon=1e-6):
    # Compute the product U_0 U_0^T
    product = np.dot(U_0, U_0.T)

    # Compute the trace of the product
    trace = np.trace(product)

    # Check if the trace is approximately 1
    trace_satisfied = np.isclose(trace, 1, atol=epsilon)

    # Check if all eigenvalues are non-negative
    eigenvalues = np.linalg.eigvalsh(product)
    eigenvalues_satisfied = np.all(eigenvalues >= -epsilon)

    return trace_satisfied, eigenvalues_satisfied

def check_constraints(YY_flat, epsilon=1e-6):
    n = int(np.sqrt(len(YY_flat)))
    YY = csr_matrix(YY_flat.reshape((n, n)))

    # Trace constraint
    trace_constraint = np.isclose(YY.trace(), 1, atol=epsilon)

    # Eigenvalue constraint
    eigenvalues = np.linalg.eigvalsh(YY.toarray())
    eigenvalue_constraint = all(eigval >= -epsilon for eigval in eigenvalues)

    return trace_constraint, eigenvalue_constraint


"""
def constraint_function(YY_flat):
    n = int(np.sqrt(len(YY_flat)))
    YY = csr_matrix(YY_flat.reshape((n, n)))

    # Define the constraints
    constraints = []
    constraints.append(np.trace(YY) - 1)  # trace(YY^T) ≤ 1

    # Check if YY^T is positive semidefinite
    eigenvalues = np.linalg.eigvalsh(YY.A)
    for eigval in eigenvalues:
        constraints.append(eigval >= 0)

    return constraints
"""
"""
def constraint_function(YY_flat):
    # Reshape YY_flat into a symmetric matrix YY^T
    n = int(np.sqrt(len(YY_flat)))
    # YY = YY_flat.reshape((n, n))
    YY = csr_matrix(YY_flat.reshape((n, n)))

    # Define the constraints
    constraints = []
    constraints.append(np.trace(YY) - 1)  # trace(YY^T) ≤ 1
    
    # Check if YY^T is positive semidefinite
    eigenvalues = np.linalg.eigvalsh(YY.toarray())
    for eigval in eigenvalues:
        constraints.append(eigval >= 0)

    return constraints
"""
"""
def constraint_function(YY_flat):
    # Reshape YY_flat into a symmetric matrix YY^T
    n = int(np.sqrt(len(YY_flat)))
    YY = YY_flat.reshape((n, n))

    # Define the constraints
    constraints = []
    constraints.append(np.trace(YY) - 1)  # trace(YY^T) ≤ 1
    
    # Check symmetry
    if not np.allclose(YY, YY.T):
        constraints.append(np.sum((YY - YY.T)**2))  # Enforce symmetry

    # Check non-negativity of diagonal elements
    constraints.extend(YY.diagonal() >= 0)

    return constraints
"""
def redefine_C(X):
    # Get the shape of the sparse matrix X
    n, _ = X.shape
    e = csr_matrix(np.ones(n))
    # Initialize C as a sparse matrix with the same shape as X
    C = -e.dot(e.T)
    
    # Set non-zero values in C based on the edges
    #for i, j in edges:
    #    C[i, j] = -1
    #    C[j, i] = -1  # Assuming an undirected graph
    
    return C

import numpy as np

def calculate_U(X, s):
    # calculating U ∈ (n x s) where UUT a s-rank approximation of X
    # Perform eigen decomposition on X
    eigvals, eigvecs = np.linalg.eigh(X)
    
    # Take the square root of the largest s eigenvalues
    largest_eigvals = eigvals[-s:]
    U = np.sqrt(largest_eigvals) * eigvecs[:, -s:]
    
    return U

# --------------------------------------------------------------

def hallar(U_0, p_0, epsilon_c, epsilon_p, beta, rho, lambda_0):
    
    # defining initial iterates
    t = 1
    U_t = U_0
    p_t = p_0

    # threshold
    epsilon = min(epsilon_c, epsilon_p**2 * beta / 6)

    # solving U_t iteratively until stopping criterion is met (maximum residual)
    while True:
        # defining lagrangian for p_t
        def L_beta(X):
            return np.trace(C.dot(X)) + np.dot(p_t.T, A(X) - b) + beta/2 * np.linalg.norm(A(X) - b) ** 2
            # OR
            # return np.trace(C.dot(X)) + np.dot(p_t.T, AX_minus_b) + beta/2 * np.linalg.norm(AX_minus_b) ** 2
            # How define when maximization problem?

        def L_beta_cp(X):
            # Define the objective function using CVXPY operations
            objective = cp.trace(C @ X) + cp.sum(cp.multiply(p_t.T, A_cp(X) - b)) + beta/2 * cp.norm(A_cp(X) - b)**2
            return objective
        
        def L_beta_scipy(YY_flat):
            n = int(np.sqrt(len(YY_flat)))
            # X = YY_flat.reshape((n, n))
            X = csr_matrix(YY_flat.reshape((n, n)))
            C_sparse = csr_matrix(C)
            return C_sparse.dot(X).trace() + np.dot(p_t.T, A(X) - b) + beta/2 * np.linalg.norm(A(X) - b) ** 2

        """
        def constraints(X):
            # ensuring X is positive semidefinite and has unit trace?
            trace_constraint = np.trace(X) - 1
            psd_constraint = -X  # X should be negative semidefinite
            return trace_constraint, psd_constraint

        def L_beta_with_constraints(X):
            obj_term = L_beta(X)
            trace_constraint, psd_constraint = constraints(X)
            penalty_term = np.linalg.norm(trace_constraint) ** 2 + np.linalg.norm(psd_constraint) ** 2
            return obj_term + penalty_term
        """

        # defining initial Y_0 by warm start
        Y_0 = U_t

        trace_satisfied, eigenvalues_satisfied = check_constraints_U(Y_0)
        print("Trace constraint satisfied:", trace_satisfied)
        print("Eigenvalues constraint satisfied:", eigenvalues_satisfied)

        # calling HLR with initial Y_0, giving next iterate U_t
        
        # U_t = hlr(Y_0, L_beta, lambda_0, epsilon, rho)
        # value, U_t = opt_solver(L_beta_cp, Y_0)
        
        # ----------------- scipy optimize ------------------------------------
        # optimizing in full rank to then create s-rank approximation of output
        # Initial guess for YY^T
        YY_initial_guess = Y_0.dot(Y_0.T)

        # Flatten the initial guess for optimization
        YY_initial_guess_flat = YY_initial_guess.flatten()

        # Define the bounds for YY_flat (optional)
        #bounds = [(-np.inf, np.inf)] * len(YY_initial_guess_flat)

        # Define the constraints using dictionaries
        constraints = {'type': 'ineq', 'fun': lambda x: constraint_function(x)}

        # Optimize the objective function subject to the constraints
        result = minimize(L_beta_scipy, YY_initial_guess_flat, constraints=constraints) # , bounds=bounds

        # Retrieve the optimal solution
        optimal_solution_x_flat = result.x
        optimal_solution_x = optimal_solution_x_flat.reshape(YY_initial_guess.shape)
        #print("X")
        #print(optimal_solution_x)
        #print("U")

        
        #print(U_t)
        #print("UUT")
        #print(U_t.dot(U_t.T))
        #optimal_solution = csr_matrix(optimal_solution_flat.reshape(YY_initial_guess.shape))

        # Evaluate constraints for the optimal solution YY
        trace_constraint, eigenvalue_constraint = check_constraints(optimal_solution_x_flat)

        # Check if constraints are satisfied
        if trace_constraint <= 0 and eigenvalue_constraint:
            print("Constraints are satisfied.")
        else:
            print("Constraints are not satisfied.")

        """ CHECK """
        """ Byt till allt i x-domän eller projicera UUT tillbaka på spektraplex """
        U_t = calculate_U(optimal_solution_x, Y_0.shape[1])

        # -----------------------------------------------------------------------

        # U_t = gradient_descent(L_beta, Y_0, .1, p_t, beta)

        # Updating Lagrangian multiplier with the violation of constraints (and penalty parameter beta)
        """ CHECK """
        p_t = p_t + beta * (A(U_t.dot(U_t.T)) - b)

        # stopping condition ||A(UU.T)-b|| < epsilon_p
        if np.linalg.norm(A(U_t.dot(U_t.T)) - b) < epsilon_p: # ord='fro' ?
            break

        # safety break
        if t > 10:
            break
        
        """ vilken L_beta? """
        print("Iteration {}:".format(t), np.linalg.norm(A(U_t.dot(U_t.T)) - b), L_beta_scipy(U_t.dot(U_t.T)))

        t = t + 1

    # Minimal eigenvalue computation
    theta_t = theta_tilde(U_t.dot(U_t.T), p_t, beta)

    return U_t.dot(U_t.T), p_t, theta_t


X, p_t, theta_t = hallar(
    # initial points
        U_0         = initialize_U(n, 1),
        p_0         = np.zeros(m),
    # tolerance pair
        epsilon_c   = .05,
        epsilon_p   = .05, 
    # penalty parameter
        beta        = 1, 
    # ADAP-AIPP parameters
        rho         = 1,
        lambda_0    = 1
    )