from tqdm import tqdm
import networkx as nx
import numpy as np
from time import time
from src.preprocessing.preprocessing import *
import os
from src.shape.shape import shape

from concurrent.futures import ProcessPoolExecutor, as_completed


def main():
    data_path = "data/nonrigid3d/"
    preprocessing_export_path = "data/preprocessing/"
    networks = []
    mat_paths = []
    print("Loading Shapes...")
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".mat"):
            mat_path = os.path.join(data_path, file)
            G = load_and_construct_matrix(mat_path)
            networks.append(G)
            mat_paths.append(mat_path)

    fps_samples = []
    if not os.path.exists(preprocessing_export_path):
        os.makedirs(preprocessing_export_path)

    # Sequential
    # for i in tqdm(range(len(networks))):
    #     network = networks[i]
    #     mat_path = mat_paths[i]
    #     fps, distance_matrix = preprocess_matrix(
    #         network, mat_path, preprocessing_export_path
    #     )
    #     fps_samples.append(fps)

    batch_preprocess_matrix(
        networks, mat_paths, preprocessing_export_path, num_workers=10
    )

    # batch_size = 8
    # batch = []
    # num_batches = len(networks) // batch_size + 1
    # curr_batch = 1
    # for i in range(len(networks)):
    #     network = networks[i]
    #     mat_path = mat_paths[i]
    #     batch.append((network, mat_path))

    #     if len(batch) == batch_size:
    #         print(f"Processing batch {curr_batch}/{num_batches}")
    #         futures = []
    #         with ProcessPoolExecutor() as executor:
    #             for network, mat_path in batch:
    #                 future = executor.submit(
    #                     preprocess_matrix, network, mat_path, preprocessing_export_path
    #                 )
    #                 futures.append(future)
    #             for future in tqdm(as_completed(futures), total=len(futures)):
    #                 fps, _ = future.result()
    #                 fps_samples.append(fps)

    #         batch = []
    # if len(batch) > 0:
    #     print(f"Processing batch {curr_batch}/{num_batches}")
    #     futures = []
    #     with ProcessPoolExecutor() as executor:
    #         for network, mat_path in batch:
    #             future = executor.submit(
    #                 preprocess_matrix, network, mat_path, preprocessing_export_path
    #             )
    #             futures.append(future)
    #         for future in tqdm(as_completed(futures), total=len(futures)):
    #             fps, _ = future.result()
    #             fps_samples.append(fps)


if __name__ == "__main__":
    main()
