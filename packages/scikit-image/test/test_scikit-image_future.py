from textwrap import dedent

import pytest

def test_skimage_future_manual_lasso_segmentation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        from skimage import data, future, io
        camera = data.camera()
        mask = future.manual_lasso_segmentation(camera) 
    """)
    selenium.run(cmd)
 
def test_skimage_future_manual_polygon_segmentation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        from skimage import data, future, io
        camera = data.camera()
        mask = future.manual_polygon_segmentation(camera)  
    """)
    selenium.run(cmd)
 