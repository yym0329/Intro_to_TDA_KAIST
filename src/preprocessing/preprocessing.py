# from shape.shape import shape
from tqdm import tqdm
import networkx as nx
import numpy as np
from time import time


# It takes too long, and we don't need to use it.
def construct_geodesic_distance_matrix(G):
    """Construct a geodesic distance matrix from a graph. Use dijkstra algorithm to find the shortest path between two nodes.
    The network G is weighted and undirected graph representation of a 3D object's shape.

    Args:
        G (nx.Graph): weighted and undirected graph representation of a 3D object's shape

    Returns:
        np.array: geodesic distance matrix
    """
    n = G.number_of_nodes()
    start = time()
    num_samples = 5 if n >= 5 else n
    samples = np.random.choice(n, (num_samples, 2), replace=False)
    for i in range(num_samples):
        a = samples[i]

        nx.shortest_path_length(G, a[0], a[1], weight="weight")
    end = time()
    estimated_time = (end - start) * (n * n) / (num_samples)
    print(f"Estimated time: min, {estimated_time / 60}, sec, {estimated_time % 60}")
    distance_matrix = np.zeros((n, n))

    for i in tqdm(range(n)):
        for j in range(n):
            distance_matrix[i, j] = nx.shortest_path_length(G, i, j, weight="weight")

    return distance_matrix
