import numpy as np
from scipy.io import savemat, loadmat


class Shape:
    def __init__(self, name, dm):
        """
        3D object with as a weighted undirected graph

        Args:
            name (String): name of the shape
            dm (np.array): distance matrix of the shape
        """
        self.name = name

        if type(dm) != np.ndarray:
            raise TypeError("dm must be a numpy array")
        self.dm = dm

    def save_to_mat(self, path):
        savemat(path, {"name": self.name, "dm": self.dm})

    def load_from_mat(self, path):
        mat_data = loadmat(path)
        self.name = mat_data["name"][0]
        self.dm = mat_data["dm"]
