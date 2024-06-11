"""
Defining SDP for Maximum stable set:
max {ee^T•X : X_ij=0, ij ∈ E, tr(X)=1, X ∈ S^n}
"""

import numpy as np
import networkx as nx
from scipy.io import mmread

def import_graph_from_mtx(file_path):
    # Read graph adjacency matrix from .mtx file
    adjacency_matrix = mmread(file_path)
    
    # Convert to NetworkX graph and get node and edge set
    graph = nx.convert_matrix.from_scipy_sparse_array(adjacency_matrix)
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    print(len(nodes), len(edges))
    # Find a maximum stable set (independent set)
    max_stable_set = nx.maximal_independent_set(graph)
    
    return nodes, edges, len(max_stable_set)

def create_random_graph(num_nodes, num_edges):
    # Create a random graph
    G = nx.gnm_random_graph(num_nodes, num_edges)
    
    # Ensure the nodes are numbered starting from 1 for consistency with your original example
    G = nx.relabel_nodes(G, lambda x: x + 1)
    
    # Get the list of nodes and edges
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    # Find a maximum stable set (independent set)
    max_stable_set = nx.maximal_independent_set(G)

    return nodes, edges, len(max_stable_set)

def define_vars(nodes, edges): # A function to define variables of MSS problem
    n = len(nodes)
    m = len(edges)
    e = np.ones(n)
    C = - np.outer(e, e)
    b = np.zeros(m)
    return n, m, C, b

def A(X, m, nodes, edges):
    AX = np.empty(m)
    for k, (i, j) in enumerate(edges): 
        " check edges from nx if (1,7) and (7,1) "
        " factr 2 in A or adjoint? "
        # Assuming X is a sparse matrix representation of the adjacency matrix
        AX[k] = X[nodes.index(i), nodes.index(j)]
    return AX

def A_adjoint(v, n, nodes, edges):
    M = np.zeros((n, n))
    for k, (i, j) in enumerate(edges):
        M[nodes.index(i), nodes.index(j)] = v[k]
        # M[nodes.index(j), nodes.index(i)] = v[k]
    return M

def check_constraints(Y_flat, n, epsilon=1e-2): # Check constraints with set margin epsilon
    # Trace constraint
    trace_constraint = np.isclose(np.sum(np.square(Y_flat)), 1, atol=epsilon)   

    return trace_constraint

"""
Check A and A* functions. 
""" 
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