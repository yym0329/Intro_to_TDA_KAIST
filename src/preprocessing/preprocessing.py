# from shape.shape import shape
from tqdm import tqdm
import networkx as nx
import numpy as np
from time import time
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from src.shape.shape import Shape


def preprocess_matrix(network, mat_path, preprocessing_export_path):
    network
    fps = farthest_first_sampling(network, k=200, exact=False)

    distance_matrix = np.zeros((len(fps), len(fps)))
    for j in range(len(fps)):
        for k in range(len(fps)):
            distance_matrix[j][k] = nx.shortest_path_length(
                network, fps[j], fps[k], weight="weight"
            )
    shape_name = mat_path.split("/")[-1].split(".")[0]
    shape_export_path = os.path.join(
        preprocessing_export_path, "pre_geodesic_" + shape_name + ".mat"
    )
    shape_obj = Shape(shape_name, distance_matrix)
    shape_obj.save_to_mat(shape_export_path)

    print("Shape " + shape_name + " saved to " + shape_export_path)
    return fps, distance_matrix


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


def construct_euclidean_distance_matrix(G):
    n = G.number_of_nodes()
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(
                G.nodes[i]["pos"] - G.nodes[j]["pos"]
            )
    return distance_matrix


def farthest_first_sampling(G, k, verbose=False, metric="geodesic"):
    """
    Farthest first sampling, also known as farthest first traversal.
    the distance from a point to a set is defined as the minimum of the pairwise distances to points in the set.
    Args:
        G (nx.Graph): weighted and undirected graph representation of a 3D object's shape
        k (Integer): The number of sample nodes to be selected
        verbose (Boolean): Whether to show the progress bar
        metric (String): The metric to be used for the distance calculation. Should be "geodesic" or "euclidean"

    Returns:
        list: list of selected nodes
        np.array: geodesic distance matrix

    """

    selected_nodes = []
    n = G.number_of_nodes()
    # Select the first node randomly
    selected_nodes.append(np.random.randint(n))
    if metric == "geodesic":
        length_matrix = construct_geodesic_distance_matrix(G, weight="weight")
    elif metric == "euclidean":
        length_matrix = construct_euclidean_distance_matrix(G)
    else:
        raise ValueError("metric should be geodesic or euclidean")

    if not verbose:
        iterator = range(k - 1)
    else:
        iterator = tqdm(range(k - 1))
    for i in iterator:
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
                farthest_distance = distance_to_set
                farthest_node = j

        selected_nodes.append(farthest_node)
        # not_selected_nodes.remove(farthest_node)

    return selected_nodes, length_matrix


def preprocess_matrix(network, mat_path, preprocessing_export_path, metric="geodesic"):
    network
    fps, dG = farthest_first_sampling(network, k=200, metric=metric)

    distance_matrix = np.zeros((len(fps), len(fps)))

    for j in range(len(fps)):
        for k in range(len(fps)):
            distance_matrix[j][k] = dG[fps[j]][fps[k]]

    coordinates = np.zeros((len(fps), 3))
    for j in range(len(fps)):
        coordinates[j] = network.nodes[fps[j]]["pos"]

    shape_name = mat_path.split("/")[-1].split(".")[0]
    shape_export_path = os.path.join(
        preprocessing_export_path, f"pre_{metric}_" + shape_name + ".mat"
    )
    shape_obj = Shape(shape_name, distance_matrix, coordinates)
    shape_obj.save_to_mat(shape_export_path)

    print("Shape " + shape_name + " saved to " + shape_export_path)
    return fps, distance_matrix


def batch_preprocess_matrix(
    networks, mat_paths, preprocessing_export_path, num_workers=8, metric="geodesic"
):
    num_batches = len(networks) // num_workers + 1
    batch = []
    tqdm_bar = tqdm(total=num_batches, desc="Preprocessing batches")

    fps_samples = []
    for i in range(len(networks)):
        network = networks[i]
        mat_path = mat_paths[i]
        batch.append((network, mat_path))

        if len(batch) == num_workers:
            futures = []
            with ProcessPoolExecutor() as executor:
                for network, mat_path in batch:
                    future = executor.submit(
                        preprocess_matrix,
                        network,
                        mat_path,
                        preprocessing_export_path,
                        metric=metric,
                    )
                    futures.append(future)
                for future in as_completed(futures):
                    fps, _ = future.result()
                    fps_samples.append(fps)
            batch = []
            tqdm_bar.update(1)
    if len(batch) > 0:
        futures = []
        with ProcessPoolExecutor() as executor:
            for network, mat_path in batch:
                future = executor.submit(
                    preprocess_matrix, network, mat_path, preprocessing_export_path
                )
                futures.append(future)
            for future in as_completed(futures):
                fps, _ = future.result()
                fps_samples.append(fps)
        tqdm_bar.update(1)
    tqdm_bar.close()
    return fps_samples
