from textwrap import dedent
import pytest

def test_skimage_draw_bezier_curve(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.draw import bezier_curve
        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = bezier_curve(1, 5, 5, -2, 8, 8, 2)
        img[rr, cc] = 1
    """)
    selenium.run(cmd)

def test_skimage_draw_circle_perimeter(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.draw import circle_perimeter
        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = circle_perimeter(4, 4, 3)
        img[rr, cc] = 1
    """)
    selenium.run(cmd)

def test_skimage_draw_disk(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #disk
        import numpy as np
        from skimage.draw import disk
        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = disk((4, 4), 5)
        img[rr, cc] = 1
    """)
    selenium.run(cmd)

def test_skimage_draw_ellipse(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #ellipse
        import numpy as np
        from skimage.draw import ellipse
        img = np.zeros((10, 12), dtype=np.uint8)
        rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
        img[rr, cc] = 1
    """)
    selenium.run(cmd)

def test_skimage_draw_line(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #line
        import numpy as np
        from skimage.draw import line
        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = line(1, 1, 8, 8)
        img[rr, cc] = 1
    """)
    selenium.run(cmd)

def test_skimage_draw_polygon(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #polygon
        import numpy as np
        from skimage.draw import polygon
        img = np.zeros((10, 10), dtype=np.uint8)
        r = np.array([1, 2, 8])
        c = np.array([1, 7, 4])
        rr, cc = polygon(r, c)
        img[rr, cc] = 1
    """)
    selenium.run(cmd)

def test_skimage_draw_polygon2mask(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #polygon2mask
        import numpy as np
        from skimage.draw import line, polygon2mask
        image_shape = (128, 128)
        polygon = np.array([[60, 100], [100, 40], [40, 40]])
        mask = polygon2mask(image_shape, polygon)
        mask.shape
    """)
    selenium.run(cmd)

def test_skimage_draw_rectangle(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #rectangle
        import numpy as np
        from skimage.draw import rectangle
        img = np.zeros((5, 5), dtype=np.uint8)
        start = (1, 1)
        extent = (3, 3)
        rr, cc = rectangle(start, extent=extent, shape=img.shape)
        img[rr, cc] = 1
    """)
    selenium.run(cmd)

def test_skimage_draw_set_color(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #set_color
        import numpy as np
        from skimage.draw import line, set_color
        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = line(1, 1, 20, 20)
        set_color(img, (rr, cc), 1)
    """)
    selenium.run(cmd)
