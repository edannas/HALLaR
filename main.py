""" 
min {C•UUT : A(UUT)=b, ||U||≤1, U ∈ R^{n×r}}

eigenvalue check unnecessary

TODO
- clean HLR
- check A_adjoint function
- check utils: initialize_X, calculate_Y

- transform Y solution to selection of edges? 
- why and how is L approximating MSS?
- selection of s?

- remove gradient_descent?
- Hur formulera trace constraint? Kvadrat istället för abs. Olikhet ist. för likhet?
- apply timing function to check total time but also to identify areas for optimization
- plot convergence curve
- check minimal eigenvalue computation theta_tilde, why?
- selection of parameters?
- check which solver is used in scipy.optimize.minimze
- pass derivatives to optimizer
- Store all iterations and plot Y-vector and objective function (lagrangian)
- LANCElOT method for selecting adaptive beta?
- Check if approx optimal for the SDP, using duality gap? (also, compare to nx function (approximate))
  Plot fluctuations in nx function to highlight not an exact number. Boxplot between this and HALLaR for a few runs.
  (On a really small graph check with known MSS)
- We dont necesarrily need THE optimal solution because of relaxation, but if almost it would be good 
- Compute gradient using adjoint instead of finite difference, but check correctness with FD
- 15-20 pages for report
- refer functions to places in report
- resolve global variables
- compare ineq/eq-constraints (trace)
- compare built in gradient with provided gradient
- np.sum(np.square( -> np.norm(, ord = "fro")
- In scipy optimize: which minimize alg. is used? print output


FOR REPORT:
- describe computation to avoid storing X in lagrangian (see image from meeting 23/5) C*X = (CY)*Y^T
- comment reformulation of trace constraint as frobenius norm constraint? Do we need to square the frobenius norm again for it to be smooth?
- trace(YYT) -> fr_norm(Y)**2
"""

# ----------------- IMPORTS AND SETUP -----------------
import numpy as np
from utils import *
from scipy.optimize import minimize, NonlinearConstraint
import time
# from HLR import hlr
# import cvxpy as cp

# -- Import problem definition for maximum stable set --
# Functions: A, A*, constraint_function, check_constraints
#            import_graph_from_mtx, create_small_graph, create_large_graph, define_vars
from MSS_SDP import *

# nodes, edges, max_stable_set = create_graph() # 20 nodes, 23 edges
# nodes, edges, max_stable_set = create_random_graph(200, 400)

#nodes, edges, max_stable_set = read_clq_file("graphs/C125.9.clq.txt")
#print(len(nodes), len(edges))
# nodes, edges, max_stable_set = create_random_graph(250, 500)
graph = "small2"
file_path = "graphs/{}.pickle".format(graph)
# file_path = "graphs/G11.mtx" # 800 nodes, 1600 edges

import pickle

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
exit()
# nodes, edges, max_stable_set = import_graph_from_mtx(file_path)
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

"""
def compute_gradient(Y_flat, A, A_adjoint, C, p_t, q, beta, nodes):
    n = len(nodes)
    s = int(len(Y_flat) / n)
    Y = Y_flat.reshape((n, s))

    # Compute the gradient of the Lagrangian
    AX_b = A(Y.dot(Y.T), m, nodes, edges) - b
    adjoint_term = A_adjoint(AX_b, n, nodes, edges)
    grad_YY = C + adjoint_term + beta * A_adjoint(AX_b, n, nodes, edges)

    # Compute the gradient with respect to Y
    grad_Y = 2 * grad_YY.dot(Y)

    return grad_Y.flatten()
"""
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
    iterations = []  # Example iteration numbers
    objective_values = []  # Example objective function values
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
        # ----------------- cvxpy optimize -----------------
        # For this method all variables and functions have to be redefined using cvxpy functions
        """
        # Define optimization variable and objective function
        n, s = Y_t.shape
        Z = cp.Variable((n, n), symmetric=True)
        objective = cp.Minimize(L_beta(Z))

        # Define constraints and formulate optimization problem
        constraints = [cp.trace(Z) <= 1, Z >> 0]
        problem = cp.Problem(objective, constraints)

        # Solve and retrieve value/solution
        problem.solve()
        optimal_value = problem.value
        optimal_solution = Z.value

        optimal_solution_Y = calculate_Y(optimal_solution)
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
        
        #print(1)
        # Optimize the objective function subject to the trace constraint
        result = minimize(L_beta_scipy, Y_initial_flat, constraints = nlc, 
                          args=(A, A_adjoint, C, p_t, q, beta, nodes),
                          jac = compute_gradient,
                          method = "trust-constr") # SLSQP, COBYLA
                          #options={'disp': True}) # , bounds=bounds
        #print(2)
        
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

        iterations.append(t)
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

    return Y_t, p_t, theta_t, L_beta_scipy(optimal_solution_y_flat), iterations, objective_values, constraint_violations

beta = 500
rank = 1

start_time = time.time()
Y_t, p_t, theta_t, L_value, iterations, objective_values, constraint_violations = hallar(
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

print("Value:", -L_value, "AX-b:", constraint_violations[-1], "No. It:", iterations[-1], "comp_time:", comp_time, "Nx-value:", max_stable_set)

#n = 3
#m = 2
#nodes, edges, max_stable_set = create_random_graph(n, m)
#v = np.array([1,2])
#M = np.array([[1,2,3],[1,2,3],[1,2,3]])
#IP1 = (A(M, m, nodes, edges)).dot(v)
#IP2 = np.sum(M * (A_adjoint(v, n, nodes, edges)))
#print(M, v)
#print(IP1)
#print(IP2)

import matplotlib.pyplot as plt

def plot_metrics(iterations, objective_values, constraint_violations, reference_value):
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

    # Add legends
    #fig.tight_layout()  # Adjust layout to make room for both y-labels
    #ax1.legend(loc='upper right')
    #ax2.legend(loc='upper right')

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

# Example usage
#iterations = list(range(1, 11))  # Example iteration numbers
#objective_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # Example objective function values
#constraint_violations = [1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0]  # Example constraint violations
reference_value = max_stable_set  # Example reference value for the objective function

plot_metrics(iterations, objective_values, constraint_violations, reference_value)
