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

        if not G.has_edge(*edge1):
            G.add_edge(*edge1, weight=weight_1)
        else:
            print("not included repetitoin")
            G.add_edge(*edge1, weight=weight_1)
        if not G.has_edge(*edge2):
            G.add_edge(*edge2, weight=weight_2)
        else:
            print("not included repetitoin")
            G.add_edge(*edge2, weight=weight_1)

        if not G.has_edge(*edge3):
            G.add_edge(*edge3, weight=weight_3)
        else:
            print("not included repetitoin")
            G.add_edge(*edge3, weight=weight_3)

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
def construct_geodesic_distance_matrix(G, weight="weight", use_parallel=False):
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

        nx.shortest_path_length(G, a[0], a[1], weight=weight)
    end = time()
    estimated_time = (end - start) * (n * n) / (num_samples)
    print(f"Average time per calculation: {(end-start) / num_samples} sec")
    print(f"Estimated time: min, {estimated_time / 60}, sec, {estimated_time % 60}")
    distance_matrix = np.zeros((n, n))

    if not use_parallel:
        print("Use single thread")
        print(f"The current weight setting is {weight}")
        for i in tqdm(range(n)):
            for j in range(n):
                distance_matrix[i, j] = nx.shortest_path_length(G, i, j, weight=weight)

    else:
        futures = []
        counter = 0
        max_num = 500
        ind_array = np.zeros((max_num, 2))
        with ProcessPoolExecutor() as executor:
            for i in range(n):
                for j in range(n):
                    ind_array[counter, 0] = i
                    ind_array[counter, 1] = j
                    counter += 1
                    if counter == max_num:
                        futures.append(
                            executor.submit(shortest_path_length, G, ind_array)
                        )
                        counter = 0
                        ind_array = np.zeros((max_num, 2))
                    # futures.append(executor.submit(shortest_path_length, G, i, j))
        if counter != 0:
            futures.append(
                executor.submit(shortest_path_length, G, ind_array[:counter])
            )

        done = 0
        for future in as_completed(futures):
            source, target, l = future.result()
            distance_matrix[source, target] = l
            done += 1
            if done % 100 == 0:
                print(
                    f"Done {done*max_num} out of {n*n} calculations. {done*max_num/(n*n)*100}%"
                )

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
    # Select the rest of the nodes
    distance_memo = dict()
    not_selected_nodes = list(range(n))
    if not exact:
        num_nodes_to_search = int(n * num_nodes)
    for i in tqdm(range(k - 1)):
        # Find the farthest node from the selected nodes
        farthest_node = -1
        farthest_distance = -1

        if exact:
            search_set = range(n)

        else:
            cands = (
                num_nodes_to_search
                if num_nodes_to_search < len(not_selected_nodes)
                else len(not_selected_nodes)
            )
            # print("The number of candidates", cands)
            search_set = np.random.choice(not_selected_nodes, cands, replace=False)

        for j in search_set:
            if j in selected_nodes:
                continue
            distances = []
            for x in selected_nodes:
                search_string = f"{x},{j}" if x < j else f"{j},{x}"
                if search_string in distance_memo:
                    distances.append(distance_memo[search_string])
                else:
                    distance = nx.shortest_path_length(G, j, x, weight="weight")
                    distance_memo[search_string] = distance

                    distances.append(distance)
            distance_to_set = np.min(distances)
            if distance_to_set > farthest_distance:
                farthest_node = j
                farthest_distance = distance

        selected_nodes.append(farthest_node)
        not_selected_nodes.remove(farthest_node)

    return selected_nodes
