from textwrap import dedent

import pytest

def test_skimage_filters_apply_hysteresis_threshold(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.filters import apply_hysteresis_threshold
        image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
        apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    """)
    selenium.run(cmd)

def test_skimage_filters_gabor(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #gabor
        from skimage.filters import gabor
        from skimage import data
        image = data.coins()
        # detecting edges in a coin image
        filt_real, filt_imag = gabor(image, frequency=0.6)
    """)
    selenium.run(cmd)

def test_skimage_filters_gaussian(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #gaussian
        from skimage.filters import gaussian
        from skimage.data import astronaut
        image = astronaut()
        filtered_img = gaussian(image, sigma=1, multichannel=True)
    """)
    selenium.run(cmd)

def test_skimage_filters_median(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #median
        from skimage import data
        from skimage.morphology import disk
        from skimage.filters import median
        img = data.camera()
        med = median(img, disk(5))
    """)
    selenium.run(cmd)

def test_skimage_filters_prewitt(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #prewitt
        from skimage import data
        from skimage import filters
        camera = data.camera()
        edges = filters.prewitt(camera)
    """)
    selenium.run(cmd)


def test_skimage_filters_rank_order(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #rank_order
        from skimage.filters import rank_order
        import numpy as np
        a = np.array([[1, 4, 5], [4, 4, 1], [5, 1, 1]])
        rank_order(a)
    """)
    selenium.run(cmd)


def test_skimage_filters_roberts(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #roberts
        from skimage import data
        camera = data.camera()
        from skimage import filters
        edges = filters.roberts(camera)
    """)
    selenium.run(cmd)


def test_skimage_filters_sobel(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #sobel
        from skimage import data
        from skimage import filters
        camera = data.camera()
        edges = filters.sobel(camera)
    """)
    selenium.run(cmd)

def test_skimage_filters_threshold_isodata(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #threshold_isodata
        from skimage import filters
        from skimage.data import coins
        image = coins()
        thresh = filters.threshold_isodata(image)
        binary = image > thresh
    """)
    selenium.run(cmd)

def test_skimage_filters_threshold_li(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #threshold_li
        from skimage import filters
        from skimage.data import camera
        image = camera()
        thresh = filters.threshold_li(image)
        binary = image > thresh
    """)
    selenium.run(cmd)

def test_skimage_filters_threshold_local(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #threshold_local
        from skimage.data import camera
        from skimage import filters
        image = camera()[:50, :50]
        binary_image1 = image > filters.threshold_local(image, 15, 'mean')
        func = lambda arr: arr.mean()
        binary_image2 = image > filters.threshold_local(image, 15, 'generic',
                                                param=func)
    """)
    selenium.run(cmd)

def test_skimage_filters_threshold_mean(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #threshold_mean
        from skimage.data import camera
        from skimage import filters
        image = camera()
        thresh = filters.threshold_mean(image)
        binary = image > thresh
    """)
    selenium.run(cmd)

def test_skimage_filters_threshold_multiotsu(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #threshold_multiotsu
        from skimage import filters
        from skimage.color import label2rgb
        from skimage import data
        import numpy as np
        image = data.camera()
        thresholds = filters.threshold_multiotsu(image)
        regions = np.digitize(image, bins=thresholds)
        regions_colorized = label2rgb(regions)
    """)
    selenium.run(cmd)

def test_skimage_filters_threshold_otsu(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #threshold_otsu
        from skimage import filters
        from skimage.data import camera
        image = camera()
        thresh = filters.threshold_otsu(image)
        binary = image <= thresh
    """)
    selenium.run(cmd)

def test_skimage_filters_try_all_thresholds(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #try_all_thresholds
        from skimage.data import text
        from skimage import filters
        fig, ax = filters.try_all_threshold(text(), figsize=(10, 6), verbose=False)
    """)
    selenium.run(cmd)

def test_skimage_filters_unsharp_mask(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #unsharp_mask
        from skimage import filters
        import numpy as np
        array = np.ones(shape=(5,5), dtype=np.uint8)*100
        array[2,2] = 120
        np.around(filters.unsharp_mask(array, radius=0.5, amount=2),2)
    """)
    selenium.run(cmd)

def test_skimage_filters_unsharp_mask(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #window
        from skimage.filters import window
        w = window('hann', (512, 512))
    """)
    selenium.run(cmd)