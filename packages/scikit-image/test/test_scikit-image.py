from textwrap import dedent

import pytest

def test_skimage_import(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage.measure import label, regionprops
        from skimage import measure
        from skimage.color import label2rgb

        # The imports below don't work, all give the same error.
        from skimage.filters import threshold_otsu # (temporarily) replaced by bl_threshold_otsu
        from skimage.morphology import closing, square
        from skimage.segmentation import find_boundaries, chan_vese, watershed
        from skimage.segmentation import mark_boundaries

    """)
    selenium.run(cmd)


def test_skimage_img_as_type_functions(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage import data, img_as_bool, img_as_float, img_as_float32, img_as_float64, img_as_ubyte, img_as_uint

        image = img_as_bool(data.coins())
        image = img_as_float(data.coins())
        image = img_as_float32(data.coins())
        image = img_as_float64(data.coins())
        image = img_as_ubyte(data.coins())
        image = img_as_uint(data.coins())
    """)
    selenium.run(cmd)



def test_skimage_label(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.measure import label, regionprops
        x = np.eye(3).astype(int)

        label_img = label(x, connectivity=1)
        props = regionprops(label_img)
    """)
    selenium.run(cmd)

def test_skimage_morphology(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
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


def test_skimage_boundaries(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.segmentation import find_boundaries, chan_vese, watershed
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

def test_skimage_watershed(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.segmentation import watershed

        x, y = np.indices((80, 80))
        x1, y1, x2, y2 = 28, 28, 44, 52
        r1, r2 = 16, 20
        mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
        mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
        image = np.logical_or(mask_circle1, mask_circle2)
        from scipy import ndimage as ndi
        distance = ndi.distance_transform_edt(image)
        from skimage.feature import peak_local_max
        local_maxi = peak_local_max(distance, labels=image,
                                    footprint=np.ones((3, 3)),
                                    indices=False)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=image)
    """)
    selenium.run(cmd)

def test_skimage_mark_and_find_boundaries(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import data
        from skimage import color
        from skimage import morphology
        from skimage import segmentation

        # Input data
        img = data.immunohistochemistry()

        # Compute a mask
        lum = color.rgb2gray(img)
        mask = morphology.remove_small_holes(
            morphology.remove_small_objects(
                lum < 0.7, 500),
            500)

        mask = morphology.opening(mask, morphology.disk(3))

        # SLIC result
        slic = segmentation.slic(img, n_segments=200, start_label=1)
        np.shape(slic)
        segmentation.mark_boundaries(img, slic)

    """)
    selenium.run(cmd)