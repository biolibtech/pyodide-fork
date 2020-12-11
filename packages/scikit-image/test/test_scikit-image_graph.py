from textwrap import dedent
import pytest


def test_skimage_graph_route_through_array(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #route_through_array
        import numpy as np
        from skimage.graph import route_through_array

        image = np.array([[1, 3], [10, 12]])
        # Forbid diagonal steps
        route_through_array(image, [0, 0], [1, 1], fully_connected=False)
        # Now allow diagonal steps: the path goes directly from start to end
        route_through_array(image, [0, 0], [1, 1])
        # Cost is the sum of array values along the path (16 = 1 + 3 + 12)
        route_through_array(image, [0, 0], [1, 1], fully_connected=False,
        geometric=False)
        # Larger array where we display the path that is selected
        image = np.arange((36)).reshape((6, 6))
    """)
    selenium.run(cmd)