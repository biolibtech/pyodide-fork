from textwrap import dedent

import pytest

def test_scipy_spatial_Delaunay_triangulations(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.spatial import Delaunay

        points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
        tri = Delaunay(points)

        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
        tri = Delaunay(points)
    """)
    selenium.run(cmd)

def test_scipy_spatial_convex_hull(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.spatial import ConvexHull
        points = np.random.rand(30, 2)   # 30 random points in 2-D
        hull = ConvexHull(points)
    """)
    selenium.run(cmd)

def test_scipy_spatial_kd_tree(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.spatial import KDTree
        import numpy as np
        points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                        [2, 0], [2, 1], [2, 2]])
        tree = KDTree(points)
        tree.query([0.1, 0.1])
    """)
    selenium.run(cmd)

