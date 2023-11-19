import numpy as np
from scipy.io import savemat


class shape:
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
