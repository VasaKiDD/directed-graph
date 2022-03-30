import os

import numpy as np

from graphs.graphs import DirectedGraph


def test_init():
    G = DirectedGraph()
    num_vertices, num_edges = G.dim()

    assert num_vertices == 0
    assert num_edges == 0


def test_add_vertices():
    vertices = np.arange(11)
    vertices[-1] = 0
    G = DirectedGraph()
    G.add_vertices(vertices)
    num_vertices, _ = G.dim()
    assert num_vertices == 10
    G.add_vertices([3, 11])
    num_vertices, _ = G.dim()
    assert num_vertices == 11


def test_add_edges():
    edges = np.random.randint(10, size=(100, 2))
    G = DirectedGraph()
    G.add_edges(edges, add_vertices=False)
    _, num_edges = G.dim()
    assert num_edges == 0
    G.add_edges(edges, add_vertices=True)
    num_vertices, num_edges = G.dim()
    assert num_edges == np.unique(edges, axis=0).shape[0]
    assert num_vertices == np.unique(edges).shape[0]


def test_get_vertices():
    vertices = np.arange(10).tolist()
    G = DirectedGraph()
    G.add_vertices(vertices)

    vertices_back = G.get_vertices()

    assert (vertices_back == vertices).all()


def test_get_edges():
    edges = np.random.randint(10, size=(100, 2))
    G = DirectedGraph()
    G.add_edges(edges)

    unique_edges = np.unique(edges, axis=0).tolist()
    unique_edges = np.array(sorted(map(tuple, unique_edges)))
    unique_edges = np.array(unique_edges)

    back_edges = np.array(G.get_edges())

    assert (back_edges == unique_edges).all()


def test_eq():
    edges = np.random.randint(10, size=(100, 2))
    vertices = np.arange(10)
    G1 = DirectedGraph(vertices, edges)
    assert G1 == G1
    vertices[0] = 15
    G2 = DirectedGraph(vertices, edges)
    assert G1 != G2


def test_add():
    vertices1 = [0, 1]
    vertices2 = [1, 2]
    edges1 = [[0, 1]]
    edges2 = [[1, 2], [1, 2]]
    G1 = DirectedGraph(vertices1, edges1)
    G2 = DirectedGraph(vertices2, edges2)
    G_test = DirectedGraph(vertices1 + vertices2, edges1 + edges2)
    G_add = G1 + G2
    assert G_test == G_add


def test_h5_serialisation():
    edges = np.random.randint(10, size=(100, 2))
    vertices = np.arange(10)
    G1 = DirectedGraph(vertices, edges)
    G1.save_to_h5(path="./tests/assets", name="test_graph")
    assert os.path.exists("./tests/assets/test_graph.hdf5")
    G2 = DirectedGraph()
    G2.load_from_h5("./tests/assets/test_graph.hdf5")
    assert G1 == G2


def test_save_out_distribution():
    edges = np.random.randint(1000, size=(100000, 2))
    vertices = np.arange(1000)
    G = DirectedGraph(vertices, edges)
    G.save_out_distribution("./tests/assets", "test_graph.png")
