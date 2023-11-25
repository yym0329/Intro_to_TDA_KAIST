import numpy as np
from scipy.io import savemat, loadmat
import ripser
import persim
import gudhi
from gudhi.wasserstein import wasserstein_distance


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
        self.dgms = self.get_VR_diagram()
        self.diameter = self.get_diameter()

    def save_to_mat(self, path):
        """
        Save the shape to a .mat file

        Args:
            path (String): path to save the shape to
        """
        savemat(path, {"name": self.name, "dm": self.dm})

    def load_from_mat(self, path):
        """
        Load the shape from a .mat file

        Args:
            path (String): path to load the shape from
        """
        mat_data = loadmat(path)
        self.name = mat_data["name"][0]
        self.dm = mat_data["dm"]

    def get_VR_diagram(self, max_dim=2):
        """
        Compute the persistence diagram of the shape with Vietoris-Rips filtration

        Args:
            max_dim (int, optional): Maximum dimension to compute persistence homology. Defaults to 2.

        Returns:
            np.array: The persistence diagram of the shape
        """
        self.dgms = ripser.ripser(self.dm, maxdim=max_dim, distance_matrix=True)["dgms"]
        return self.dgms

    def plot_diagram(self):
        """
        Plot the persistence diagram of the shape.
        """
        if self.dgms is None:
            self.get_VR_diagram()
        persim.plot_diagrams(self.dgms, show=True)

    def get_diameter(self):
        """
        Compute the diameter of the shape

        Returns:
            np.float: The diameter of the shape
        """
        return np.max(self.dm)


def load_from_mat(path):
    """
    Load a shape from a .mat file

    Args:
        path (String): path to the .mat file

    Returns:
        Shape: The shape object
    """
    mat_data = loadmat(path)
    name = mat_data["name"][0]
    dm = mat_data["dm"]
    return Shape(name, dm)


def dDiam(shape1, shape2):
    """
    Diameter distance between two shapes.

    Args:
        shape1 (Shape): The first shape object
        shape2 (Shape): The second shape object

    Returns:
        np.float: The diameter distance between the two shapes
    """
    return np.abs(shape1.get_diameter() - shape2.get_diameter())


def d_E_inf(shape1, shape2):
    """
    Implementation of d_E_inf for 0,1,2 dimensional persistence diagrams.

    Args:
        shape1 (Shape): The first shape object
        shape2 (Shape): The second shape object

    Returns:
        float: The E_inf distance between the two shapes
    """
    bottleneck_distance_zeroth = gudhi.bottleneck_distance(
        shape1.dgms[0], shape2.dgms[0]
    )
    bottleneck_distance_first = gudhi.bottleneck_distance(
        shape1.dgms[1], shape2.dgms[1]
    )
    bottleneck_distance_second = gudhi.bottleneck_distance(
        shape1.dgms[2], shape2.dgms[2]
    )

    return max(
        bottleneck_distance_zeroth,
        bottleneck_distance_first,
        bottleneck_distance_second,
    )


def d_G_wasserstein(shape1, shape2, q=1):
    """
    Implementation of d_G_wasserstein for 0,1,2 dimensional persistence diagrams.

    Args:
        shape1 (Shape): The first shape object
        shape2 (Shape): The second shape object
        q (float, optional): The order of the Wasserstein distance. Defaults to 1.

    Returns:
        float: The Wasserstein distance between the two shapes
    """
    wasserstein_distance_zeroth = wasserstein_distance(
        shape1.dgms[0], shape2.dgms[0], order=q
    )
    wasserstein_distance_first = wasserstein_distance(
        shape1.dgms[1], shape2.dgms[1], order=q
    )
    wasserstein_distance_second = wasserstein_distance(
        shape1.dgms[2], shape2.dgms[2], order=q
    )

    return max(
        wasserstein_distance_zeroth,
        wasserstein_distance_first,
        wasserstein_distance_second,
    )
