""" 
min {C•UUT : A(UUT)=b, ||U||≤1, U ∈ R^{n×r}}

eigenvalue check unnecessary
trace(YYT) -> fr_norm(Y)**2

TODO
- clean HLR
- check A_adjoint function
- check utils: initialize_X, calculate_Y

- transform Y solution to selection of edges? 
- why and how is L approximating MSS?

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
"""

# ----------------- IMPORTS AND SETUP -----------------
import numpy as np
from utils import *
from scipy.optimize import minimize, NonlinearConstraint
# from HLR import hlr
# import cvxpy as cp

# -- Import problem definition for maximum stable set --
# Functions: A, A*, constraint_function, check_constraints
#            import_graph_from_mtx, create_small_graph, create_large_graph, define_vars
from MSS_SDP import *

file_path = "graphs/small.mtx"
# nodes, edges, max_stable_set = import_graph_from_mtx(file_path)
nodes, edges, max_stable_set = create_graph()
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

            # Compute X = YY^T and create compressed sparse row matrices.
            X = csr_matrix(Y.dot(Y.T)) 
            " without constructing X "
            C_sparse = csr_matrix(C) 
            " move to global "
            """
            CX -> (CY)Y^T
            storing A(X)
            """

            # Compute value of lagrangian function at X
            return C_sparse.dot(X).trace() + np.dot(p_t.T, A(X, m, nodes, edges) - b) \
                + beta/2 * np.linalg.norm(A(X, m, nodes, edges) - b) ** 2

        # Define and flatten the initial guess for optimization (warm start)
        Y_initial = Y_t
        Y_initial_flat = Y_initial.flatten()

        # Define the bounds for Y_flat?
        # bounds = [(-np.inf, np.inf)] * len(Y_initial_flat)

        # Trace constraint
        """ sum of squared entries of Y """
        con = lambda Y: np.trace(Y.reshape(n, int(len(Y)/n)).dot(Y.reshape(n, int(len(Y)/n)).T))**2 - 1
        nlc = NonlinearConstraint(con, -np.inf, 0)

        """ CHECK BELOW """
        #print(np.linalg.eigvalsh(Y_t.dot(Y_t.T)))
        
        """ which minimize alg.? print output """
        # Optimize the objective function subject to the trace constraint
        result = minimize(L_beta_scipy, Y_initial_flat, constraints = nlc)#,
                          #jac = compute_gradient, args=(A, A_adjoint, C, p_t, q, beta, nodes)) # , bounds=bounds
        
        # Retrieve the optimal solution
        optimal_solution_y_flat = result.x
        optimal_solution_y = optimal_solution_y_flat.reshape(Y_initial.shape) # create csr_matrix?
        
        " reformulate trace exp"
        print("Trace of solution:", np.round(np.trace(optimal_solution_y.dot(optimal_solution_y.T)), 5))

        # Evaluate and check constraints for the optimal solution
        trace_constraint, eigenvalue_constraint = check_constraints(optimal_solution_y_flat, n)
        
        if not (trace_constraint and eigenvalue_constraint):
            print("Constraints are not satisfied (iteration {})".format(t))
        else:
            print("Constraints are satisfied.")
            pass
        
        # Define next iterate
        Y_t = optimal_solution_y
        # -----------------------------------------------------------------------

        # Update Lagrangian multiplier (violation of constraints with penalty parameter beta)
        """ CHECK """
        " reuse A(X) "

        " compare ineq/eq-constraints (trace) "
        p_t = p_t + beta * (A(Y_t.dot(Y_t.T), m, nodes, edges) - b)

        # Stopping condition ||A(UU.T)-b|| < epsilon_p
        if np.linalg.norm(A(Y_t.dot(Y_t.T), m, nodes, edges) - b) < epsilon_p: # ord='fro' ?
            print("Stopping criterion met (||A(UU.T)-b|| = {} < {} = epsilon_p)".format(
                np.round(np.linalg.norm(A(Y_t.dot(Y_t.T), m, nodes, edges) - b), 5),
                epsilon_p))
            break

        # Safety break
        if t > max_iter:
            print("Maximum iterations reached ({})".format(max_iter))
            break
        
        print("Iteration {}: ||A(UU.T)-b|| = {}, L(Y) = {}".format(t, \
            np.round(np.linalg.norm(A(Y_t.dot(Y_t.T), m, nodes, edges) - b), 5), 
            np.round(L_beta_scipy(optimal_solution_y_flat), 5)))

        t = t + 1

    # Minimal eigenvalue computation
    theta_t = theta_tilde(Y_t.dot(Y_t.T), p_t, beta, A, A_adjoint, C, q)

    return Y_t, p_t, theta_t, L_beta_scipy(optimal_solution_y_flat)


Y_t, p_t, theta_t, L_value = hallar(
        Y_0 = calculate_Y(initialize_X(n), 1), p_0 = np.zeros(m),
        epsilon_c = .05, epsilon_p = .05, 
        beta = 50,
        rho = 1, lambda_0 = 1,
        max_iter = 100
    )


print("Maximum stable set (nx):", max_stable_set)
print("Maximum stable set (HALLaR):", np.round(-L_value, 5))
print("Trace of YY^T:", np.round(np.trace(Y_t.dot(Y_t.T)), 5))