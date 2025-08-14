import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_causal_graph(G_mm, threshold=0.5):
    """
    Create a directed graph from a causality matrix.
    
    Args:
        G_mm: numpy array of shape (5,5) representing causality matrix
        threshold: float, minimum value to consider as a causal connection
        
    Returns:
        nx.DiGraph: Directed graph representing causal relationships
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (regions)
    regions = ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5']
    G.add_nodes_from(regions)
    
    # Add edges based on causality matrix
    for i in range(5):
        for j in range(5):
            if i != j and abs(G_mm[i,j]) > threshold:
                G.add_edge(regions[i], regions[j], weight=G_mm[i,j])
    
    return G

def plot_causal_graph(G, title="Causal Network"):
    """
    Plot the causal graph with edge weights.
    
    Args:
        G: nx.DiGraph, the graph to plot
        title: str, title for the plot
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.6)
    
    # Draw edges with weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_weights,
                          edge_cmap=plt.cm.Reds, width=2,
                          arrows=True, arrowsize=20)
    
    # Add edge labels (weights)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" 
                  for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    # Add node labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage:
# G_mm = np.array([[0, 0.8, 0.2, 0.1, 0.3],
#                  [0.4, 0, 0.6, 0.2, 0.1],
#                  [0.1, 0.3, 0, 0.7, 0.2],
#                  [0.2, 0.1, 0.5, 0, 0.6],
#                  [0.3, 0.2, 0.1, 0.4, 0]])
# 
# G = create_causal_graph(G_mm)
# plot_causal_graph(G) 