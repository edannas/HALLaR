""" 
Run one time to generate sample graphs. 
""" 
import networkx as nx
import pickle

def create_random_graph(num_nodes, num_edges):
    # Create a random graph
    G = nx.gnm_random_graph(num_nodes, num_edges)
    
    # Ensure the nodes are numbered starting from 1 for consistency with your original example
    G = nx.relabel_nodes(G, lambda x: x + 1)
    return G

small1 = create_random_graph(100, 200)
small2 = create_random_graph(250, 500)
mid = create_random_graph(500, 1000)
# mid2 = create_random_graph(500, 5000)

# save graph objects sto file
# pickle.dump(small1, open('graphs/small1.pickle', 'wb'))
# pickle.dump(small2, open('graphs/small2.pickle', 'wb'))
# pickle.dump(mid, open('graphs/mid.pickle', 'wb'))
# pickle.dump(mid2, open('graphs/mid2.pickle', 'wb'))