from tqdm import tqdm
import networkx as nx
import numpy as np
from time import time
from src.preprocessing.preprocessing import *
import os
from src.shape.shape import shape


def main():
    data_path = "data/nonrigid3d/"
    preprocessing_export_path = "data/preprocessing/"
    networks = []
    mat_paths = []
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".mat"):
            mat_path = os.path.join(data_path, file)
            G = load_and_construct_matrix(mat_path)
            networks.append(G)
            mat_paths.append(mat_path)

    fps_samples = []
    if not os.path.exists(preprocessing_export_path):
        os.makedirs(preprocessing_export_path)
    for i in tqdm(range(len(networks))):
        network = networks[i]
        fps = farthest_first_sampling(network, k=10, exact=False)
        fps_samples.append(fps)
        distance_matrix = np.zeros((len(fps), len(fps)))
        for j in range(len(fps)):
            for k in range(len(fps)):
                distance_matrix[j][k] = nx.shortest_path_length(
                    network, fps[j], fps[k], weight="weight"
                )
        shape_name = mat_paths[i].split("/")[-1].split(".")[0]
        shape_export_path = os.path.join(
            preprocessing_export_path, "pre_geodesic_" + shape_name + ".mat"
        )
        shape_obj = shape(shape_name, distance_matrix)
        shape_obj.save_to_mat(shape_export_path)


if __name__ == "__main__":
    main()
