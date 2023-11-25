from tqdm import tqdm
from src.preprocessing.preprocessing import *
import os


def main():
    data_path = "data/nonrigid3d/"
    preprocessing_export_path = "data/preprocessing/euclidean/"
    networks = []
    mat_paths = []
    print("Loading Shapes...")
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".mat"):
            mat_path = os.path.join(data_path, file)
            G = load_and_construct_matrix(mat_path)
            networks.append(G)
            mat_paths.append(mat_path)
    print("Shapes loaded.")
    if not os.path.exists(preprocessing_export_path):
        os.makedirs(preprocessing_export_path)

    batch_preprocess_matrix(
        networks, mat_paths, preprocessing_export_path, num_workers=8
    )


if __name__ == "__main__":
    main()
