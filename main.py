""" 
min {C•UUT : A(UUT)=b, ||U||≤1, U ∈ R^{n×r}}
"""

# ----------------- IMPORTS AND SETUP -----------------
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.optimize import minimize, NonlinearConstraint
import time, pickle

# -- Import problem definition for maximum stable set --
# Functions: A, A*, constraint_function, check_constraints
#            import_graph_from_mtx, create_small_graph, create_large_graph, define_vars
from MSS_SDP import *

# -- Create test graphs --
# nodes, edges, max_stable_set = create_random_graph(200, 400)
graph = "small2" # small1, small2, mid
file_path = "graphs/{}.pickle".format(graph)

# load graph object from file
G = pickle.load(open(file_path, 'rb'))
nodes = list(G.nodes)
edges = list(G.edges)

# Find a maximum stable set (independent set)
max_stable_sets = []

k = 0
while k < 10:
    max_stable_sets.append(len(nx.maximal_independent_set(G)))
    k += 1 

max_stable_set = sum(max_stable_sets)/len(max_stable_sets)

print("avg:", max_stable_set, "max:", max(max_stable_sets), "min:", min(max_stable_sets))

n, m, C, b = define_vars(nodes, edges)

def q(Y, p, beta, A):
    return p + beta * (A(Y.dot(Y.T), m, nodes, edges) - b)

def theta_tilde(Y, p, beta, A, A_adjoint, C, q):
    # Compute the minimum eigenvalue of C + A(q(Y; p))
    min_eigenvalue = np.linalg.eigvalsh(C + A_adjoint(q(Y, p, beta, A), n, nodes, edges)).min()
    
    return max(-min_eigenvalue, 0)

def compute_gradient(Y_flat, A, A_adjoint, C, p_t, q, beta, nodes):
    n = len(nodes)
    s = int(len(Y_flat) / n)
    Y = Y_flat.reshape((n, s))
    grad_YY = C + A_adjoint(q(Y, p_t, beta, A), n, nodes, edges)
    grad_Y = 2 * grad_YY.dot(Y)
    return grad_Y.flatten()

# ----------------- HALLaR algorithm -----------------
def hallar(
    # initial points
        Y_0, p_0,
    # tolerance pair
        epsilon_c, epsilon_p,
    # penalty parameter
        beta, 
    # ADAP-AIPP parameters
        rho, lambda_0,
    # Maximum iterations
        max_iter
        ):
    
    # Define initial iterates
    t = 1
    Y_t = Y_0
    p_t = p_0

    # Calculate threshold for HLR
    """
    epsilon = min(epsilon_c, epsilon_p**2 * beta / 6)
    """
    
    # Store iterates for convergence analysis
    objective_values = []
    constraint_violations = []

    # Solve Y_t (minimize Lagrangian) iteratively until stopping criterion is met
    # YY^T is automatically symmetric PSD. 
    while True:
        # ----------------- HLR method -----------------
        """
        # Define lagrangian for constant p_t
        def L_beta(Y):
            return np.trace(C.dot(Y.dot(Y.T))) + np.dot(p_t.T, A(Y.dot(Y.T)) - b, m, nodes, edges) \
                + beta/2 * np.linalg.norm(A(Y.dot(Y.T)) - b, m, nodes, edges) ** 2

        U_t = hlr(Y_0, L_beta, lambda_0, epsilon, rho)
        """

        # ----------------- scipy optimize -----------------
        # Define lagrangian for constant p_t
        def L_beta_scipy(Y_flat, *args):
            # Reshape Y_flat to Y
            n = len(nodes)
            s = int(len(Y_flat) / n)
            Y = Y_flat.reshape((n, s))

            AX_b = A(Y.dot(Y.T), m, nodes, edges) - b

            # Compute value of lagrangian function at YY^T
            return C.dot(Y).dot(Y.T).trace() + np.dot(p_t.T, AX_b) \
                + beta/2 * np.linalg.norm(AX_b) ** 2

        # Define and flatten the initial guess for optimization (warm start)
        Y_initial = Y_t
        Y_initial_flat = Y_initial.flatten()

        # Define the bounds for Y_flat?
        # bounds = [(-np.inf, np.inf)] * len(Y_initial_flat)

        # Trace constraint
        con = lambda Y: np.sum(np.square(Y)) - 1
        nlc = NonlinearConstraint(con, 0, 0)

        """ CHECK BELOW """
        #print(np.linalg.eigvalsh(Y_t.dot(Y_t.T)))
        
        # Optimize the objective function subject to the trace constraint
        result = minimize(L_beta_scipy, Y_initial_flat, constraints = nlc, 
                          args=(A, A_adjoint, C, p_t, q, beta, nodes),
                          jac = compute_gradient,
                          method = "trust-constr") # SLSQP, COBYLA
                          #options={'disp': True}) # , bounds=bounds
        
        # Retrieve the optimal solution
        optimal_solution_y_flat = result.x
        optimal_solution_y = optimal_solution_y_flat.reshape(Y_initial.shape)
        
        print("Trace of solution:", np.round(np.sum(np.square(optimal_solution_y)), 5))

        # Evaluate and check constraints for the optimal solution
        trace_constraint = check_constraints(optimal_solution_y_flat, n)
        
        if not trace_constraint:
            print("Constraint not satisfied (iteration {})".format(t))
        else:
            print("Constraint satisfied.")
            pass
        
        # Define next iterate
        Y_t = optimal_solution_y
        # -----------------------------------------------------------------------

        AX_b = A(Y_t.dot(Y_t.T), m, nodes, edges) - b
        AX_b_norm = np.linalg.norm(AX_b)

        # Update Lagrangian multiplier (violation of constraints with penalty parameter beta)
        p_t = p_t + beta * AX_b
        
        opt_value = L_beta_scipy(optimal_solution_y_flat)

        objective_values.append(-opt_value)
        constraint_violations.append(AX_b_norm)

        # Stopping condition ||A(UU.T)-b|| < epsilon_p
        if AX_b_norm < epsilon_p: # ord='fro' ?
            print("Stopping criterion met (||A(UU.T)-b|| = {} < {} = epsilon_p)".format(
                np.round(AX_b_norm, 5),
                epsilon_p))
            break

        # Safety break
        if t > max_iter:
            print("Maximum iterations reached ({})".format(max_iter))
            break

        print("Iteration {}: ||A(UU.T)-b|| = {}, L(Y) = {}".format(t, \
            np.round(AX_b_norm, 5), 
            np.round(opt_value, 5)))

        t = t + 1

    # Minimal eigenvalue computation
    theta_t = theta_tilde(Y_t.dot(Y_t.T), p_t, beta, A, A_adjoint, C, q)

    return Y_t, p_t, theta_t, L_beta_scipy(optimal_solution_y_flat), objective_values, constraint_violations

beta = 500
rank = 1

start_time = time.time()
Y_t, p_t, theta_t, L_value, objective_values, constraint_violations = hallar(
        # Y_0 = calculate_Y(initialize_X(n), 2), p_0 = np.zeros(m),
        Y_0 = generate_Y(n, rank), p_0 = np.zeros(m),
        epsilon_c = .05, epsilon_p = 1e-2, 
        beta = beta,
        rho = 1, lambda_0 = 1,
        max_iter = 1000
    )
end_time = time.time()
comp_time = end_time - start_time

print("Maximum stable set (nx):", max_stable_set)
print("Maximum stable set (HALLaR):", np.round(-L_value, 5))
print("Trace of YY^T:", np.round(np.sum(np.square(Y_t.dot(Y_t.T))), 5))

print("Value:", -L_value, "AX-b:", constraint_violations[-1], "No. It:", len(constraint_violations), "comp_time:", comp_time, "Nx-value:", max_stable_set)

plot_metrics(graph, beta, rank, objective_values, constraint_violations, max_stable_set)
