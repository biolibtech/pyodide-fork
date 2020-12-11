from textwrap import dedent
import pytest


def test_skimage_future_graph_cut_normalized(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        from skimage import data, segmentation
        from skimage.future import graph
        img = data.coffee()
        labels = segmentation.slic(img)
        rag = graph.rag_mean_color(img, labels, mode='similarity')
        new_labels = graph.cut_normalized(labels, rag)
    """)
    selenium.run(cmd)

def test_skimage_future_graph_cut_threshold(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        from skimage import data, segmentation
        from skimage.future import graph
        img = data.coffee()
        labels = segmentation.slic(img)
        rag = graph.rag_mean_color(img, labels)
        new_labels = graph.cut_threshold(labels, rag, 10)
    """)
    selenium.run(cmd)

def test_skimage_future_graph_rag_boundary(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        from skimage import data, segmentation, filters, color
        from skimage.future import graph
        img = data.coffee()
        labels = segmentation.slic(img)
        edge_map = filters.sobel(color.rgb2gray(img))
        rag = graph.rag_boundary(labels, edge_map)
    """)
    selenium.run(cmd)

def test_skimage_future_graph_rag_mean_color(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        from skimage import data, segmentation
        from skimage.future import graph
        img = data.coffee()
        labels = segmentation.slic(img)
        rag = graph.rag_mean_color(img, labels)
    """)
    selenium.run(cmd)

def test_skimage_future_graph_show_rag(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        from skimage import data, segmentation
        from skimage.future import graph
        import matplotlib.pyplot as plt
        img = data.coffee()
        labels = segmentation.slic(img)
        g =  graph.rag_mean_color(img, labels)
        lc = graph.show_rag(labels, g, img)
        cbar = plt.colorbar(lc)
    """)
    selenium.run(cmd)

def test_skimage_future_graph_nx_graph(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        import networkx as nx
        G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        G = nx.Graph(name='my graph')
        e = [(1, 2), (2, 3), (3, 4)]  # list of edges
        G = nx.Graph(e)
    """)
    selenium.run(cmd)