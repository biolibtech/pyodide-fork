from textwrap import dedent
import pytest


def test_skimage_registration_active_contour(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #active_contour
        from skimage.draw import circle_perimeter
        from skimage.filters import gaussian
        from skimage.segmentation import active_contour
        import numpy as np
        img = np.zeros((100, 100))
        rr, cc = circle_perimeter(35, 45, 25)
        img[rr, cc] = 1
        img = gaussian(img, 2)
        s = np.linspace(0, 2*np.pi, 100)
        init = 50 * np.array([np.sin(s), np.cos(s)]).T + 50
        snake = active_contour(img, init, w_edge=0, w_line=1, coordinates='rc')  
        dist = np.sqrt((45-snake[:, 0])**2 + (35-snake[:, 1])**2)
    """)
    selenium.run(cmd)

def test_skimage_registration_clear_border(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #clear_border
        import numpy as np
        from skimage.segmentation import clear_border
        labels = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 1, 0, 0, 1, 0, 0, 1, 0],
                        [1, 1, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        clear_border(labels)
    """)
    selenium.run(cmd)

def test_skimage_registration_felzenzwalb(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #felzenzwalb
        from skimage.segmentation import felzenszwalb
        from skimage.data import coffee
        img = coffee()
        segments = felzenszwalb(img, scale=3.0, sigma=0.95, min_size=5)
    """)
    selenium.run(cmd)

def test_skimage_registration_find_boundaries(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #find_boundaries
        import numpy as np
        from skimage.segmentation import find_boundaries
        labels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                        [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                        [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                        [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                        [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        find_boundaries(labels, mode='thick').astype(np.uint8)
    """)
    selenium.run(cmd)

def test_skimage_registration_join_segmentation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #join_segmentation
        import numpy as np
        from skimage.segmentation import join_segmentations
        s1 = np.array([[0, 0, 1, 1],
                    [0, 2, 1, 1],
                    [2, 2, 2, 1]])
        s2 = np.array([[0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 1]])
        join_segmentations(s1, s2)
    """)
    selenium.run(cmd)

def test_skimage_registration_random_walker(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #random_walker
        from skimage.segmentation import random_walker
        import numpy as np
        np.random.seed(0)
        a = np.zeros((10, 10)) + 0.2 * np.random.rand(10, 10)
        a[5:8, 5:8] += 1
        b = np.zeros_like(a, dtype=np.int32)
        b[3, 3] = 1  # Marker for first phase
        b[6, 6] = 2  # Marker for second phase
        random_walker(a, b)
    """)
    selenium.run(cmd)

def test_skimage_registration_relabel_sequential(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #relabel_sequential
        import numpy as np
        from skimage.segmentation import relabel_sequential
        label_field = np.array([1, 1, 5, 5, 8, 99, 42])
        relab, fw, inv = relabel_sequential(label_field)
    """)
    selenium.run(cmd)

# def test_skimage_registration_slic(selenium_standalone, request):
#     selenium = selenium_standalone
#     selenium.load_package("scikit-image")
#     cmd = dedent(r"""
#         #slic
#         from skimage.segmentation import slic
#         from skimage.data import astronaut
#         img = coffee()
#         segments = slic(img, n_segments=100, compactness=10)
#     """)
#     selenium.run(cmd)
