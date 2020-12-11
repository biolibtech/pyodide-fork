from textwrap import dedent

import pytest


def test_skimage_features_adjust_gamma(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import data, exposure, img_as_float
        image = img_as_float(data.moon())
        gamma_corrected = exposure.adjust_gamma(image, 2)
        # Output is darker for gamma > 1
        image.mean() > gamma_corrected.mean()
    """)
    selenium.run(cmd)

def test_skimage_features_blob_dog(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #blob_dog
        from skimage import data, feature
        feature.blob_dog(data.coins(), threshold=.5, max_sigma=40)
    """)
    selenium.run(cmd)

def test_skimage_features_blob_doh(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #blob_doh
        from skimage import data, feature
        img = data.coins()
        feature.blob_doh(img)
    """)
    selenium.run(cmd)

def test_skimage_features_blob_log(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #blob_log
        from skimage import data, feature, exposure
        img = data.coins()
        img = exposure.equalize_hist(img)  # improves detection
        feature.blob_log(img, threshold = .3)
    """)
    selenium.run(cmd)

def test_skimage_features_canny(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #canny
        from skimage import feature
        # Generate noisy image of a square
        im = np.zeros((256, 256))
        im[64:-64, 64:-64] = 1
        im += 0.2 * np.random.rand(*im.shape)
        # First trial with the Canny filter, with the default smoothing
        edges1 = feature.canny(im)
        # Increase the smoothing for better results
        edges2 = feature.canny(im, sigma=3)
    """)
    selenium.run(cmd)

def test_skimage_features_corners(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #corners
        from skimage.feature import corner_fast, corner_peaks
        square = np.zeros((12, 12))
        square[3:9, 3:9] = 1
        square.astype(int)
        corner_peaks(corner_fast(square, 9), min_distance=1, threshold_rel=0)
    """)
    selenium.run(cmd)

def test_skimage_features_corners_harris(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #corners_harris
        from skimage.feature import corner_harris, corner_peaks
        square = np.zeros([10, 10])
        square[2:8, 2:8] = 1
        square.astype(int)
        corner_peaks(corner_harris(square), min_distance=1, threshold_rel=0)
    """)
    selenium.run(cmd)

def test_skimage_features_corner_orientations(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #corner_orientations
        from skimage.morphology import octagon
        from skimage.feature import (corner_fast, corner_peaks,
                                    corner_orientations)
        square = np.zeros((12, 12))
        square[3:9, 3:9] = 1
        square.astype(int)
        corners = corner_peaks(corner_fast(square, 9), min_distance=1,
                            threshold_rel=0)
        orientations = corner_orientations(square, corners, octagon(3, 2))
        np.rad2deg(orientations)
    """)
    selenium.run(cmd)

def test_skimage_features_haar_like_feature(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #haar_like_feature
        from skimage.feature import haar_like_feature_coord
        from skimage.feature import draw_haar_like_feature
        feature_coord, _ = haar_like_feature_coord(2, 2, 'type-4')
        image = draw_haar_like_feature(np.zeros((2, 2)),
                                    0, 0, 2, 2,
                                    feature_coord,
                                    max_n_features=1)
    """)
    selenium.run(cmd)

def test_skimage_features_greycomatrix(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #greycomatrix
        from skimage.feature import greycomatrix
        image = np.array([[0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 2, 2, 2],
                        [2, 2, 3, 3]], dtype=np.uint8)
        result = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=4)
    """)
    selenium.run(cmd)


def test_skimage_features_hessian_matrix(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #hessian_matrix
        from skimage.feature import hessian_matrix
        square = np.zeros((5, 5))
        square[2, 2] = 4
        Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc')
    """)
    selenium.run(cmd)

def test_skimage_features_hessian_matrix_eigvals(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #hessian_matrix_eigvals
        from skimage.feature import hessian_matrix, hessian_matrix_eigvals
        square = np.zeros((5, 5))
        square[2, 2] = 4
        H_elems = hessian_matrix(square, sigma=0.1, order='rc')
        hessian_matrix_eigvals(H_elems)[0]
    """)
    selenium.run(cmd)

def test_skimage_features_match_template(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=True, reason='Possibly blocked on Dispatchable support in numpy. Need version 1.16'))
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.feature import match_template
        #match_template
        template = np.zeros((3, 3))
        template[1, 1] = 1
        image = np.zeros((6, 6))
        image[1, 1] = 1
        image[4, 4] = -1
        result = match_template(image, template)
        np.round(result, 3)
    """)
    selenium.run(cmd)

def test_skimage_features_peak_local_max(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.feature import peak_local_max
        #peak_local_max
        img1 = np.zeros((7, 7))
        img1[3, 4] = 1
        img1[3, 2] = 1.5
        peak_local_max(img1, min_distance=1)
    """)
    selenium.run(cmd)


def test_skimage_features_shape_index(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #shape_index
        from skimage.feature import shape_index
        square = np.zeros((5, 5))
        square[2, 2] = 4
        s = shape_index(square, sigma=0.1)
    """)
    selenium.run(cmd)

def test_skimage_features_structure_tensor(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #structure_tensor
        from skimage.feature import structure_tensor
        square = np.zeros((5, 5))
        square[2, 2] = 1
        Axx, Axy, Ayy = structure_tensor(square, sigma=0.1)
    """)
    selenium.run(cmd)

def test_skimage_features_ORB(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        #match_descriptors
        from skimage.feature import ORB, match_descriptors
        img1 = np.zeros((100, 100))
        img2 = np.zeros_like(img1)
        np.random.seed(1)
        square = np.random.rand(20, 20)
        img1[40:60, 40:60] = square
        img2[53:73, 53:73] = square
        detector_extractor1 = ORB(n_keypoints=5)
        detector_extractor2 = ORB(n_keypoints=5)
        detector_extractor1.detect_and_extract(img1)
        detector_extractor2.detect_and_extract(img2)
        matches = match_descriptors(detector_extractor1.descriptors,
                                    detector_extractor2.descriptors)
        detector_extractor1.keypoints[matches[:, 0]]
        detector_extractor2.keypoints[matches[:, 1]]
    """)
    selenium.run(cmd)


