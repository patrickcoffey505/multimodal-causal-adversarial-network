import numpy as np
import networkx as nx
from typing import List
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

def create_causal_graph(A: np.ndarray, threshold=0, region_names: List[str] = None, scaling=False):
    """
    Create a directed graph from a connectivity matrix.
    
    Args:
        A: numpy array of shape (T xN x N)
        threshold: float, minimum value to consider as a causal connection
        
    Returns:
        nx.DiGraph: Directed graph representing causal relationships
    """
    # Create directed graph
    G = nx.DiGraph()

    A_avg = np.mean(A, axis=0)
    if scaling:
        x_mean = np.mean(A_avg)
        x_std = np.std(A_avg)
        scaled = (A_avg - x_mean) / (x_std + 1e-8)
    else:
        scaled = A_avg

    num_regions = A_avg.shape[0]
    
    # Add nodes (regions)
    if region_names is None:
        region_names = [f"Region {i}" for i in range(1, num_regions + 1)]
    G.add_nodes_from(region_names)
    
    # Add edges based on causality matrix
    for i in range(num_regions):
        for j in range(num_regions):
            if i != j and (scaled[i,j] > threshold):
                G.add_edge(region_names[i], region_names[j], weight=scaled[i,j])
    
    return G

def plot_causal_graph(G, title="Causal Network"):
    """
    Plot the causal graph with edge weights.
    
    Args:
        G: nx.DiGraph, the graph to plot
        title: str
    """
    plt.figure(figsize=(12, 12))
    # Use circular layout for clear geometric separation
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1500, alpha=0.8,
                          edgecolors='black', linewidths=2)
    
    # Create a new graph with only the stronger direction for each pair
    G_filtered = nx.DiGraph()
    G_filtered.add_nodes_from(G.nodes())
    
    # Add edges, keeping only the stronger direction
    for u, v in G.edges():
        if G.has_edge(v, u):  # If bidirectional
            if G[u][v]['weight'] > G[v][u]['weight']:
                G_filtered.add_edge(u, v, weight=G[u][v]['weight'])
            elif G[v][u]['weight'] > G[u][v]['weight']:
                G_filtered.add_edge(v, u, weight=G[v][u]['weight'])
        else:  # If unidirectional
            G_filtered.add_edge(u, v, weight=G[u][v]['weight'])
    
    # Draw edges with weights
    edge_weights = [G_filtered[u][v]['weight'] for u, v in G_filtered.edges()]
    logger.info(f"Edge weights: {edge_weights}")
    nx.draw_networkx_edges(G_filtered, pos, edge_color=edge_weights,
                          edge_cmap=plt.cm.Reds, width=2,
                          arrows=True, arrowsize=15,
                          min_source_margin=15, min_target_margin=15)
    
    # Add edge labels (weights)
    edge_labels = {(u, v): f"{G_filtered[u][v]['weight']:.2f}" 
                  for u, v in G_filtered.edges()}
    nx.draw_networkx_edge_labels(G_filtered, pos, edge_labels, font_size=9,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                                rotate=True,
                                label_pos=0.5,
                                clip_on=True)
    
    # Add node labels
    nx.draw_networkx_labels(G_filtered, pos, font_size=11, font_weight='bold')
    
    plt.title(title, fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create a sample connectivity matrix
    num_regions = 5
    np.random.seed(42)  # For reproducibility
    
    # Create a random connectivity matrix with some structure
    A = np.random.randn(10, num_regions, num_regions)  # 10 timepoints, 5x5 regions
    A = np.abs(A)  # Make all connections positive for simplicity
    
    # Create region names
    region_names = [f"Region {i+1}" for i in range(num_regions)]
    
    # Create and plot the causal graph
    G = create_causal_graph(A, threshold=0.5, region_names=region_names)
    plot_causal_graph(G, title="Sample Causal Network")
