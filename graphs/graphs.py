import os
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np


class DirectedGraph:
    def __init__(
        self,
        vertices=None,
        edges=None,
        add_vertices=True,
    ):
        self.G = {}

        if vertices is not None:
            self.add_vertices(vertices)

        if edges is not None:
            self.add_edges(edges, add_vertices)

    def add_vertices(self, vertices):
        """

        Parameters
        ----------
        vertices : one dimensional object (array, list, h5py.Dataset)
        of vertices

        Returns
        -------

        """
        assert (
            isinstance(vertices, list)
            or isinstance(vertices, np.ndarray)
            or isinstance(vertices, h5py.Dataset)
        ), "Vertices are not a list nor an ndarray"
        assert (
            len(np.array(vertices).shape) == 1
        ), "Vertices have a different dimension than one: shape {}".format(
            np.array(vertices).shape
        )
        for vertice in vertices:
            if vertice not in self.G.keys():
                self.G[vertice] = SimpleNamespace(
                    **{"id": vertice, "out": [], "inp": []}
                )

    def add_edges(self, edges, add_vertices=True):
        """

        Parameters
        ----------
        edges : 2 dimensionnal object (array, list, h5py.Dataset) of edges
        add_vertices : bool. the function adds vertices in edges if they are
        not existing in the graph.

        Returns
        -------

        """
        assert (
            isinstance(edges, list)
            or isinstance(edges, np.ndarray)
            or isinstance(edges, h5py.Dataset)
        ), "Edges are not a list nor an ndarray"
        assert (
            len(np.array(edges).shape) == 2
        ), "Edges have a different dimension than two: shape {}".format(
            np.array(edges).shape
        )
        for vertice1, vertice2 in edges:
            if vertice1 not in self.G.keys() and add_vertices:
                self.G[vertice1] = SimpleNamespace(
                    **{"id": vertice1, "out": [], "inp": []}
                )
            elif vertice1 not in self.G.keys() and not add_vertices:
                continue
            if vertice2 not in self.G.keys() and add_vertices:
                self.G[vertice2] = SimpleNamespace(
                    **{"id": vertice1, "out": [], "inp": []}
                )
            elif vertice2 not in self.G.keys() and not add_vertices:
                continue
            if vertice2 not in self.G[vertice1].out:
                self.G[vertice1].out.append(vertice2)
            if vertice1 not in self.G[vertice2].inp:
                self.G[vertice2].inp.append(vertice1)

    def dim(self):
        """

        Returns a tuple ot int (number of vertices, number of edges)
        -------

        """
        num_edges = sum([len(self.G[v].out) for v in self.G.keys()])
        return (len(self.G), num_edges)

    def __str__(self):
        v, e = self.dim()
        mess = "Total Vertices : {}\nTotal Edges : {}".format(v, e)
        return mess

    def __add__(self, other):
        edges = other.get_edges()
        vertices = other.get_vertices()
        res = DirectedGraph()
        res.G = self.G.copy()
        res.add_vertices(vertices)
        res.add_edges(edges)
        return res

    def __eq__(self, other):
        if self.dim() != other.dim():
            return False
        edges = self.get_edges()
        vertices = self.get_vertices()
        edges2 = other.get_edges()
        vertices2 = other.get_vertices()
        if not (np.array(edges) == np.array(edges2)).all():
            return False
        if not (np.array(vertices) == np.array(vertices2)).all():
            return False

        return True

    def __getitem__(self, item):
        """

        Parameters
        ----------
        item : identifier of the node in the graph.

        Returns a dictionarry of the node
        -------

        """
        return self.G[item]

    def get_vertices(self):
        """

        Returns one dimensional array of vertices
        -------

        """
        return np.sort(list(self.G.keys()))

    def get_edges(self):
        """

        Returns double sorted 2 dimensional array of edges
        -------

        """
        edges = []
        for v in self.G.keys():
            edges += [[v, to] for to in self.G[v].out]
        edges = np.array(sorted(map(tuple, edges)))
        return np.array(edges)

    def save_to_h5(self, path="./", name="directed_graph"):
        """
        Parameters
        ----------
        path : string path to folder to save
        name : string name of the graph

        Returns
        -------
        """
        file = os.path.join(path, name) + ".hdf5"
        edges = self.get_edges()
        with h5py.File(file, "w") as f:
            f.create_dataset("vertices", data=np.array(list(self.G.keys())))
            f.create_dataset("edges", data=np.array(edges))

    def load_from_h5(self, path):
        """
        Parameters
        ----------
        path : path to hdf5 file.
        """
        with h5py.File(path, "r") as f:
            vertices = f["vertices"]
            self.add_vertices(vertices)
            edges = f["edges"]
            self.add_edges(edges)

    def vertex_out_distribution(self):
        """
        Gives the out degree distribution of the graph

        Returns tuple of probalities and associated degree
        -------

        """
        degrees = np.array([len(self.G[v].out) for v in self.G.keys()])
        probs, bins = np.histogram(
            degrees, bins=np.unique(degrees), density=True
        )
        return probs, bins

    def save_out_distribution(self, path, fig_name):
        """
        plots the out degree distribution of
        -------

        """
        probs, bins = self.vertex_out_distribution()

        plt.stairs(probs, bins, fill=True)
        plt.xlabel("Out Degree k")
        plt.ylabel("probability p(k)")
        plt.title("Out Degree Distribution")

        plt.savefig(os.path.join(path, fig_name))
