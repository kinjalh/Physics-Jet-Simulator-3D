import numpy as np
import math
import random
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(format='%(message)s', level=logging.INFO)


class Parton(object):
    """
    Represents a parton as a node in a binary tree (contains its own information and references
    to the left and right child)
    """

    def __init__(self, data, left_child=None, right_child=None):
        """
        Initializes a parton with the given data and children. Within this module, a parton's
        data stores a 3-D row vector representing the parton's momentum.
        :param data: stores the momentum vector of this parton
        :param left_child: reference to this parton's left child. Default value is None.
        :param right_child: reference to this parton's right child. Default value is None.
        """
        self.data = data
        self.left_child = left_child
        self.right_child = right_child


class Visualizer3D(object):
    """
    Creates a 3D visualization of a tree representing parton splits. The visualization is a plot
    on 3D axes.
    """

    def __init__(self):
        """
        Initializes a visualizer that plots 3D figures
        """
        fig = plt.figure(figsize=(10, 10))
        self._axes = fig.add_subplot(111, projection='3d')
        self._axes.set_xlabel('X')
        self._axes.set_ylabel('Y')
        self._axes.set_zlabel('Z')

    def create_graph(self, node, x_0, y_0, z_0):
        """
        Creates a 3-D plot of the vectors in the tree with current node as root. Starts plotting
        the vectors relative to an initial (x, y, z) of (x_0, y_0, z_0). Each vector starts where
        its parent vector ends.
        :param node: the root of the tree of vectors to graph
        :param x_0: The x coordinate at which the first vector starts
        :param y_0: The y coordinate at which the first vector starts
        :param z_0: The z coordinate at which the first vector starts
        :return: None
        """
        if node is not None:
            x_end = x_0 + node.data[0, 0]
            y_end = y_0 + node.data[0, 1]
            z_end = z_0 + node.data[0, 2]
            self._axes.plot([x_0, x_end], [y_0, y_end], [z_0, z_end])
            self.create_graph(node.left_child, x_end, y_end, z_end)
            self.create_graph(node.right_child, x_end, y_end, z_end)

    def visualize(self):
        """
        Displays the visualization on screen
        :return:
        """
        plt.show()


def split(p_0, z, theta_0, theta, phi):
    """
    Splits the given 3-D momentum vector of a parton into 2 momentum vectors, representing the
    splitting of the parton
    :param p_0: the 1x3 dimensional row vector representing the momentum of the initial parton
    :param z: the ratio of the energy of one of the resultant partons (after the split) to the
    initial parton's energy
    :param theta_0: the angle the initial parton is at when projected onto the xy plane
    :param theta: the angle of each of the resultant partons with respect to the initial parton
    :param phi: the azimuthal angle (angle determining z coordinate)
    :return: a tuple of the 2 momentum vectors obtained by splitting the initial parton
    """
    mag = np.sqrt(np.sum(np.power(p_0, 2)))
    mag_rad = mag * z
    mag_f = mag * (1 - z)
    theta_rad = theta_0 + theta
    theta_f = theta_0 - theta
    phi_rad = phi
    phi_f = phi + math.pi
    logging.info('theta_0 = %.2f, theta = %.2f, phi = %.2f', theta_0 * 180 / math.pi,
                 theta * 180 / math.pi, phi * 180 / math.pi)
    p_rad = np.array([[mag_rad * math.cos(theta_rad), mag_rad * math.sin(theta_rad),
                       mag_rad * math.sin(phi_rad)]])
    p_f = np.array([[mag_f * math.cos(theta_f), mag_f * math.sin(theta_f),
                     mag_f * math.sin(phi_f)]])
    np.set_printoptions(precision=2)
    logging.info('split %s into %s and %s', p_0, p_rad, p_f)
    return p_rad, p_f


def gen_z_val():
    """
    Generates a random value of z (ratio for energies of components of split parton). Can range
    from 0 to 1
    :return: a randomly generated value of z
    """
    while True:
        num = random.random()
        f = 1 / (1 + num)
        if random.random() <= f:
            return num


def gen_theta_val():
    """
    Generates a random value for theta, which can range from 0 to pi / 2
    :return: a randomly generated theta value
    """
    while True:
        num = random.random() * math.pi / 2
        f = 1 / (1 + num)
        if random.random() <= f:
            return num


def gen_phi_val():
    """
    Generates a random value of phi (azimuthal angle), which can range from 0 to pi / 2
    :return: a random value of pi
    """
    return random.random() * math.pi / 2


def create_tree(p_0, theta_0, max_layers, curr_layer):
    """
    Creates a tree of momentum vectors starting with p_0 as the root. Each momentum vector
    branches off into 2 momentums, thus creating a binary tree.
    :param p_0: The initial momentum vector
    :param theta_0: The angle of the initial vector when projected onto the xy plane
    :param max_layers: The max number of layers the tree should reach
    :param curr_layer: The current layer (when calling to create a new tree, this should be 0)
    :return: The root parton (which is a node) of the tree
    """
    if curr_layer >= max_layers:
        return None
    else:
        z = gen_z_val()
        theta = gen_theta_val()
        phi = gen_phi_val()
        p_rad, p_f = split(p_0, z, theta_0, theta, phi)
        curr = Parton(p_0)
        curr.left_child = create_tree(p_rad, theta_0 + theta, max_layers, curr_layer + 1)
        curr.right_child = create_tree(p_f, theta_0 - theta, max_layers, curr_layer + 1)
        return curr


if __name__ == '__main__':
    p_0 = np.array([[100, 100, 100]])
    p_1 = np.array([[-100, -100, -100]])

    root_0 = create_tree(p_0, math.pi / 4, 4, 0)
    root_1 = create_tree(p_1, math.pi * 5 / 4, 4, 0)

    vis = Visualizer3D()
    vis.create_graph(root_0, 0, 0, 0)
    vis.create_graph(root_1, 0, 0, 0)
    vis.visualize()
