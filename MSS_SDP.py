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

def read_clq_file(filename):
    G = nx.Graph()
    with open(filename, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.startswith('c'):
                continue
            # Read problem line to get number of vertices and edges
            elif line.startswith('p'):
                parts = line.split()
                num_vertices = int(parts[2])
                num_edges = int(parts[3])
                G.add_nodes_from(range(1, num_vertices + 1))
            # Read edge lines and add edges to the graph
            elif line.startswith('e'):
                parts = line.split()
                u = int(parts[1])
                v = int(parts[2])
                G.add_edge(u, v)

    nodes = list(G.nodes)
    edges = list(G.edges)
    max_stable_set = nx.maximal_independent_set(G)

    return nodes, edges, len(max_stable_set)

"""
def create_graph(): # A function to create a small sample graph
    
    # Define nodes and edges
    # nodes = list(range(1, 6))
    # edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]

    nodes = list(range(1, 21))
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
        (17, 18), (18, 19), (19, 20), (20, 1),  # Ring structure
        (2, 8), (5, 15), (10, 18)  # Additional edges for complexity
    ]

    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Find a maximum stable set (independent set)
    max_stable_set = nx.maximal_independent_set(G)

    return nodes, edges, len(max_stable_set)
"""

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