from textwrap import dedent

import pytest


def test_skimage_exposure_adjust_gamma(selenium_standalone, request):
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

def test_skimage_exposure_cumulative_distribution(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #cumulative_distribution
        import numpy as np
        from skimage import data, exposure, img_as_float
        image = img_as_float(data.camera())
        hi = exposure.histogram(image)
        cdf = exposure.cumulative_distribution(image)
        np.alltrue(cdf[0] == np.cumsum(hi[0])/float(image.size))
    """)
    selenium.run(cmd)

def test_skimage_exposure_histogram(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #histogram
        import numpy as np
        from skimage import data, exposure, img_as_float
        image = img_as_float(data.camera())
        np.histogram(image, bins=2)
        exposure.histogram(image, nbins=2)
    """)
    selenium.run(cmd)

def test_skimage_exposure_is_low_contrast(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #is_low_contrast
        import numpy as np
        from skimage.exposure import is_low_contrast
        image = np.linspace(0, 0.04, 100)
        is_low_contrast(image)
        image[-1] = 1
        is_low_contrast(image)
        is_low_contrast(image, upper_percentile=100)
    """)
    selenium.run(cmd)

def test_skimage_exposure_rescale_intensity(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #rescale_intensity
        import numpy as np
        from skimage.exposure import rescale_intensity
        image = np.array([51, 102, 153], dtype=np.uint8)
        rescale_intensity(image)
    """)
    selenium.run(cmd)

