from textwrap import dedent
import pytest


def test_skimage_measure_block_reduce(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #block_reduce
        import numpy as np
        from skimage.measure import block_reduce
        image = np.arange(3*3*4).reshape(3, 3, 4)
        block_reduce(image, block_size=(3, 3, 1), func=np.mean)
    """)
    selenium.run(cmd)

def test_skimage_measure_find_contours(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #find_contours
        import numpy as np
        from skimage.measure import find_contours
        a = np.zeros((3, 3))
        a[0, 0] = 1
        find_contours(a, 0.5)
    """)
    selenium.run(cmd)

def test_skimage_measure_label(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #label
        from skimage.measure import label
        import numpy as np
        x = np.eye(3).astype(int)
        label(x, connectivity=1)
    """)
    selenium.run(cmd)

def test_skimage_measure_moments(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #moments
        from skimage.measure import moments
        import numpy as np
        image = np.zeros((20, 20), dtype=np.double)
        image[13:17, 13:17] = 1
        M = moments(image)
        centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    """)
    selenium.run(cmd)

def test_skimage_measure_moments_hu(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #moments_hu
        from skimage.measure import moments
        import numpy as np
        image = np.zeros((20, 20), dtype=np.double)
        image[13:17, 13:17] = 0.5
        image[10:12, 10:12] = 1
        mu = moments_central(image)
        nu = moments_normalized(mu)
        moments_hu(nu)
    """)
    selenium.run(cmd)

def test_skimage_measure_perimeter(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #perimeter
        from skimage import data, util
        from skimage.measure import perimeter
        # coins image (binary)
        img_coins = data.coins() > 110
        # total perimeter of all objects in the image
        perimeter(img_coins, neighbourhood=4) 
    """)
    selenium.run(cmd)

def test_skimage_measure_moments_hu(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #profile_line
        from skimage.measure import profile_line
        import numpy as np
        x = np.array([[1, 1, 1, 2, 2, 2]])
        img = np.vstack([np.zeros_like(x), x, x, x, np.zeros_like(x)])
        profile_line(img, (2, 1), (2, 4))
    """)
    selenium.run(cmd)

def test_skimage_measure_moments_hu(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #regionprops
        from skimage import data, util
        from skimage.measure import label, regionprops
        img = util.img_as_ubyte(data.coins()) > 110
        label_img = label(img, connectivity=img.ndim)
        props = regionprops(label_img)
        # centroid of first labeled object
        props[0].centroid
        # centroid of first labeled object
        props[0]['centroid']
    """)
    selenium.run(cmd)

def test_skimage_measure_moments_hu(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #CircleModel
        import numpy as np
        from skimage.measure import CircleModel
        t = np.linspace(0, 2 * np.pi, 25)
        xy = CircleModel().predict_xy(t, params=(2, 3, 4))
        model = CircleModel()
        model.estimate(xy)
        tuple(np.round(model.params, 5))
        res = model.residuals(xy)
        np.abs(np.round(res, 9))
    """)
    selenium.run(cmd)

def test_skimage_measure_moments_hu(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #LineModelND
        import numpy as np
        from skimage.measure import LineModelND
        x = np.linspace(1, 2, 25)
        y = 1.5 * x + 3
        lm = LineModelND()
        lm.estimate(np.array([x, y]).T)
        tuple(np.round(lm.params, 5))
        res = lm.residuals(np.array([x, y]).T)
        np.abs(np.round(res, 9))
        np.round(lm.predict_y(x[:5]), 3)
        np.round(lm.predict_x(y[:5]), 3)
    """)
    selenium.run(cmd)