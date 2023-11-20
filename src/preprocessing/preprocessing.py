# from shape.shape import shape
from tqdm import tqdm
import networkx as nx
import numpy as np
from time import time
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_and_construct_matrix(mat_path):
    """
    Load a .mat file and construct a graph edges weighted with euclidean distance from it.

    Args:
        mat_path (String): path to the .mat file

    Returns:
        nx.Graph: weighted and undirected graph representation of a 3D object's shape
    """
    # Load the .mat file

    mat_data = loadmat(mat_path)
    surface_data = mat_data["surface"]

    # Extract vertices and triangle indices
    X = surface_data["X"][0, 0].flatten()
    Y = surface_data["Y"][0, 0].flatten()
    Z = surface_data["Z"][0, 0].flatten()
    vertices = np.column_stack((X, Y, Z))
    triangles = surface_data["TRIV"][0, 0] - 1  # Adjust for 0-based indexing in Python

    G = nx.Graph()
    for i, vertex in enumerate(vertices):
        G.add_node(i, pos=vertex)

    for triangle in triangles:
        edge1 = (triangle[0], triangle[1])
        edge2 = (triangle[0], triangle[2])
        edge3 = (triangle[1], triangle[2])

        weight_1 = np.linalg.norm(vertices[edge1[0]] - vertices[edge1[1]])
        weight_2 = np.linalg.norm(vertices[edge2[0]] - vertices[edge2[1]])
        weight_3 = np.linalg.norm(vertices[edge3[0]] - vertices[edge3[1]])

        G.add_edge(edge1[0], edge1[1], weight=weight_1)
        G.add_edge(edge2[0], edge2[1], weight=weight_2)
        G.add_edge(edge3[0], edge3[1], weight=weight_3)

    return G


def shortest_path_length(G, src_tartget_array, weight="weight"):
    if src_tartget_array.shape[1] != 2:
        raise ValueError("The shape of src_tartget_array should be (N,2)")
    results = []
    for i in range(src_tartget_array.shape[0]):
        source = src_tartget_array[i, 0]
        target = src_tartget_array[i, 1]
        l = nx.shortest_path_length(G, source, target, weight=weight)
        results.append((source, target, l))
    return (src_tartget_array, results)


# It takes too long, and we don't need to use it.
def construct_geodesic_distance_matrix(G, weight="weight"):
    """Construct a geodesic distance matrix from a graph. Use dijkstra algorithm to find the shortest path between two nodes.
    The network G is weighted and undirected graph representation of a 3D object's shape.

    Args:
        G (nx.Graph): weighted and undirected graph representation of a 3D object's shape

    Returns:
        np.array: geodesic distance matrix
    """
    n = G.number_of_nodes()

    lengths = nx.all_pairs_dijkstra_path_length(G, weight=weight)
    distance_matrix = np.zeros((n, n))
    for source, dict in lengths:
        for target, l in dict.items():
            distance_matrix[source, target] = l
    return distance_matrix


def farthest_first_sampling(G, k, exact=False, num_nodes=0.5):
    """
    Farthest first sampling, also known as farthest first traversal.
    the distance from a point to a set is defined as the minimum of the pairwise distances to points in the set.
    Args:
        G (nx.Graph): weighted and undirected graph representation of a 3D object's shape
        k (Integer): The number of sample nodes to be selected
        exact (Boolean): If True, use the exact algorithm, otherwise use the approximate algorithm.
    """

    selected_nodes = []
    n = G.number_of_nodes()
    # Select the first node randomly
    selected_nodes.append(np.random.randint(n))
    length_matrix = construct_geodesic_distance_matrix(G, weight="weight")
    for i in tqdm(range(k - 1)):
        # Find the farthest node from the selected nodes
        farthest_node = -1
        farthest_distance = -1

        for j in range(n):
            if j in selected_nodes:
                continue
            distances = []
            for selected_node in selected_nodes:
                distances.append(length_matrix[j, selected_node])
            distance_to_set = np.min(distances)
            if distance_to_set > farthest_distance:
                farthest_node = j

        selected_nodes.append(farthest_node)
        # not_selected_nodes.remove(farthest_node)

    return selected_nodes
