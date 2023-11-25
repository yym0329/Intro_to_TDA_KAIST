import numpy as np
from scipy.io import savemat, loadmat
import ripser
import persim


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
        self.dgms = None

    def save_to_mat(self, path):
        savemat(path, {"name": self.name, "dm": self.dm})

    def load_from_mat(self, path):
        mat_data = loadmat(path)
        self.name = mat_data["name"][0]
        self.dm = mat_data["dm"]

    def get_VR_diagram(self, max_dim=2):
        self.dgms = ripser.ripser(self.dm, maxdim=max_dim, distance_matrix=True)["dgms"]
        return self.dgms

    def plot_diagram(self):
        if self.dgms is None:
            self.get_VR_diagram()
        persim.plot_diagrams(self.dgms, show=True)


def load_from_mat(path):
    mat_data = loadmat(path)
    name = mat_data["name"][0]
    dm = mat_data["dm"]
    return Shape(name, dm)
