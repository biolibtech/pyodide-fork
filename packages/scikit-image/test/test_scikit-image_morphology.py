from textwrap import dedent
import pytest


def test_skimage_measure_area_closing_and_opening(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #area_closing_and_opening
        import numpy as np
        from skimage.morphology import area_closing, area_opening
        w = 12
        x, y = np.mgrid[0:w,0:w]
        f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
        f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120
        f[9:10,9:11] = 100; f[10,10] = 100
        f = f.astype(np.int)
        closed = area_closing(f, 8, connectivity=1)
        open = area_opening(f, 8, connectivity=1)
    """)
    selenium.run(cmd)

def test_skimage_measure_black_tophat(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #black_tophat
        import numpy as np
        from skimage.morphology import square, black_tophat
        dark_on_grey = np.array([[7, 6, 6, 6, 7],
                                [6, 5, 4, 5, 6],
                                [6, 4, 0, 4, 6],
                                [6, 5, 4, 5, 6],
                                [7, 6, 6, 6, 7]], dtype=np.uint8)
        black_tophat(dark_on_grey, square(3))
    """)
    selenium.run(cmd)

def test_skimage_measure_closing(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #closing
        import numpy as np
        from skimage.morphology import square, closing
        broken_line = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [1, 1, 0, 1, 1],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]], dtype=np.uint8)
        closing(broken_line, square(3))
    """)
    selenium.run(cmd)

def test_skimage_measure_diameter_closing(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #diameter_closing
        import numpy as np
        from skimage.morphology import diameter_closing
        w = 12
        x, y = np.mgrid[0:w,0:w]
        f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
        f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120
        f[9:10,9:11] = 100; f[10,10] = 100
        f = f.astype(np.int)
        closed = diameter_closing(f, 3, connectivity=1)
    """)
    selenium.run(cmd)

def test_skimage_measure_dilation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #dilation
        import numpy as np
        from skimage.morphology import square, dilation
        bright_pixel = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]], dtype=np.uint8)
        dilation(bright_pixel, square(3))
    """)
    selenium.run(cmd)

def test_skimage_measure_erosion(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #erosion
        import numpy as np
        from skimage.morphology import square, erosion
        bright_square = np.array([[0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 1, 1, 1, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0]], dtype=np.uint8)
        erosion(bright_square, square(3))
    """)
    selenium.run(cmd)

def test_skimage_measure_flood(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #flood
        import numpy as np
        from skimage.morphology import flood, flood_fill
        image = np.zeros((4, 7), dtype=int)
        image[1:3, 1:3] = 1
        image[3, 0] = 1
        image[1:3, 4:6] = 2
        image[3, 6] = 3
        flood_fill(image, (1, 1), 5)
        mask = flood(image, (1, 1))
        image_flooded = image.copy()
        image_flooded[mask] = 5
    """)
    selenium.run(cmd)

def test_skimage_measure_local_maxima(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #local_maxima
        import numpy as np
        from skimage.morphology import local_maxima, local_minima
        image = np.zeros((4, 7), dtype=int)
        image[1:3, 1:3] = 1
        image[3, 0] = 1
        image[1:3, 4:6] = 2
        image[3, 6] = 3
        local_maxima(image)
        local_minima(image)
    """)
    selenium.run(cmd)

def test_skimage_measure_max_tree(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #max_tree
        import numpy as np
        from skimage.morphology import max_tree
        image = np.array([[15, 13, 16], [12, 12, 10], [16, 12, 14]])
        P, S = max_tree(image, connectivity=2)
    """)
    selenium.run(cmd)

def test_skimage_measure_medial_axis(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #medial_axis
        import numpy as np
        from skimage.morphology import medial_axis
        square = np.zeros((7, 7), dtype=np.uint8)
        square[1:-1, 2:-2] = 1
        medial_axis(square).astype(np.uint8)
    """)
    selenium.run(cmd)

def test_skimage_measure_reconstruction(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #reconstruction
        import numpy as np
        from skimage.morphology import reconstruction
        x = np.linspace(0, 4 * np.pi)
        y_mask = np.cos(x)
        y_seed = y_mask.min() * np.ones_like(x)
        y_seed[0] = 0.5
        y_seed[-1] = 0
        y_rec = reconstruction(y_seed, y_mask)
    """)
    selenium.run(cmd)

def test_skimage_measure_remove_small_holes(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #remove_small_holes
        import numpy as np
        from skimage import morphology
        a = np.array([[1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0]], bool)
        b = morphology.remove_small_holes(a, 2)
    """)
    selenium.run(cmd)

def test_skimage_measure_skeletonize(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #skeletonize
        import numpy as np
        from skimage.morphology import skeletonize
        X, Y = np.ogrid[0:9, 0:9]
        ellipse = (1./3 * (X - 4)**2 + (Y - 4)**2 < 3**2).astype(np.uint8)
        skel = skeletonize(ellipse)
    """)
    selenium.run(cmd)

def test_skimage_measure_watershed(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #watershed
        import numpy as np
        x, y = np.indices((80, 80))
        x1, y1, x2, y2 = 28, 28, 44, 52
        r1, r2 = 16, 20
        mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
        mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
        image = np.logical_or(mask_circle1, mask_circle2)
        from scipy import ndimage as ndi
        distance = ndi.distance_transform_edt(image)
        from skimage.feature import peak_local_max
        from skimage.morphology import watershed
        local_maxi = peak_local_max(distance, labels=image,
                                    footprint=np.ones((3, 3)),
                                    indices=False)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=image)  
    """)
    selenium.run(cmd)