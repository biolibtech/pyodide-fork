from textwrap import dedent

import pytest

def test_skimage_rank_autolevel(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #autolevel
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import autolevel
        img = data.camera()
        auto = autolevel(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_enhance_contrast(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #enhance_contrast
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import enhance_contrast
        img = data.camera()
        out = enhance_contrast(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_entropy(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #entropy
        from skimage import data
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
        img = data.camera()
        ent = entropy(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_equalize(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        #equalize
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import equalize
        img = data.camera()
        equ = equalize(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_geometric_mean(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    selenium.load_package("networkx")
    cmd = dedent(r"""
        #geometric_mean
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import geometric_mean
        img = data.camera()
        avg = geometric_mean(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_gradient(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #gradient
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import gradient
        img = data.camera()
        out = gradient(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_majority(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #majority
        from skimage import data
        from skimage.filters.rank import majority
        from skimage.morphology import disk
        img = data.camera()
        maj_img = majority(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_maximum(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #maximum
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import maximum
        img = data.camera()
        out = maximum(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_mean(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #mean
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import mean
        img = data.camera()
        avg = mean(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_mean_bilateral(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #mean_bilateral
        import numpy as np
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import mean_bilateral
        img = data.camera().astype(np.uint16)
        bilat_img = mean_bilateral(img, disk(20), s0=10,s1=10)
    """)
    selenium.run(cmd)

def test_skimage_rank_median(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #median
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import median
        img = data.camera()
        med = median(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_noise_filter(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #noise_filter
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import noise_filter
        img = data.camera()
        out = noise_filter(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_pop(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #pop
        import numpy as np
        from skimage.morphology import square
        import skimage.filters.rank as rank
        img = 255 * np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)
        rank.pop(img, square(3))
    """)
    selenium.run(cmd)

def test_skimage_rank_subtract_mean(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #subtract_mean
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import subtract_mean
        img = data.camera()
        out = subtract_mean(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_rank_threshold(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #threshold
        import numpy as np
        from skimage.morphology import square
        from skimage.filters.rank import threshold
        img = 255 * np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)
        threshold(img, square(3))
    """)
    selenium.run(cmd)

def test_skimage_rank_tophat(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #tophat
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters.rank import tophat
        img = data.camera()
        out = tophat(img, disk(5)) 
    """)
    selenium.run(cmd)

def test_skimage_rank_windowed_histogram(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #windowed_histogram
        from skimage import data
        from skimage.filters.rank import windowed_histogram
        from skimage.morphology import disk
        img = data.camera()
        hist_img = windowed_histogram(img, disk(5))
    """)
    selenium.run(cmd)






