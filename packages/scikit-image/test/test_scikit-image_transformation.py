from textwrap import dedent
import pytest


def test_skimage_transform_downscale_local_mean(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #downscale_local_mean
        from skimage.transform import downscale_local_mean
        import numpy as np
        a = np.arange(15).reshape(3, 5)
        downscale_local_mean(a, (2, 3))
    """)
    selenium.run(cmd)

def test_skimage_transformation_transform(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import transform
        # estimate transformation parameters
        src = np.array([0, 0, 10, 10]).reshape((2, 2))
        dst = np.array([12, 14, 1, -20]).reshape((2, 2))
        tform = transform.estimate_transform('similarity', src, dst)
        np.allclose(tform.inverse(tform(src)), src)
        from skimage import data
        image = data.camera()
        transform.warp(image, inverse_map=tform.inverse) 
        tform2 = transform.SimilarityTransform(scale=1.1, rotation=1,
            translation=(10, 20))
        tform3 = tform + tform2
        np.allclose(tform3(src), tform2(tform(src)))
    """)
    selenium.run(cmd)

def test_skimage_transformation_frt2(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.transform import frt2
        SIZE = 59
        img = np.tri(SIZE, dtype=np.int32)
        f = frt2(img)
    """)
    selenium.run(cmd)

def test_skimage_transformation_hough_circle(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.transform import hough_circle
        from skimage.draw import circle_perimeter
        img = np.zeros((100, 100), dtype=np.bool_)
        rr, cc = circle_perimeter(25, 35, 23)
        img[rr, cc] = 1
        try_radii = np.arange(5, 50)
        res = hough_circle(img, try_radii)
        ridx, r, c = np.unravel_index(np.argmax(res), res.shape)
        r, c, try_radii[ridx]
    """)
    selenium.run(cmd)

def test_skimage_transformation_hough_circle_peaks(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import transform, draw
        from skimage.transform import hough_circle, hough_circle_peaks
        img = np.zeros((120, 100), dtype=int)
        radius, x_0, y_0 = (20, 99, 50)
        y, x = draw.circle_perimeter(y_0, x_0, radius)
        img[x, y] = 1
        hspaces = transform.hough_circle(img, radius)
        accum, cx, cy, rad = hough_circle_peaks(hspaces, [radius,])
    """)
    selenium.run(cmd)

def test_skimage_transformation_hough_ellipse(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.transform import hough_ellipse
        from skimage.draw import ellipse_perimeter
        img = np.zeros((25, 25), dtype=np.uint8)
        rr, cc = ellipse_perimeter(10, 10, 6, 8)
        img[cc, rr] = 1
        result = hough_ellipse(img, threshold=8)
    """)
    selenium.run(cmd)

def test_skimage_transformation_hough_line(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.transform import hough_line
        from skimage.draw import line
        img = np.zeros((100, 150), dtype=bool)
        img[30, :] = 1
        img[:, 65] = 1
        img[35:45, 35:50] = 1
        for i in range(90):
            img[i, i] = 1
        img += np.random.random(img.shape) > 0.95
        out, angles, d = hough_line(img)
        img = np.zeros((100, 150), dtype=bool)
        img[30, :] = 1
        img[:, 65] = 1
        img[35:45, 35:50] = 1
        rr, cc = line(60, 130, 80, 10)
        img[rr, cc] = 1
        img += np.random.random(img.shape) > 0.95
        out, angles, d = hough_line(img)
        fix, axes = plt.subplots(1, 2, figsize=(7, 4))
        axes[0].imshow(img, cmap=plt.cm.gray)
        axes[0].set_title('Input image')
        axes[1].imshow(
            out, cmap=plt.cm.bone,
            extent=(np.rad2deg(angles[-1]), np.rad2deg(angles[0]), d[-1], d[0]))
        axes[1].set_title('Hough transform')
        axes[1].set_xlabel('Angle (degree)')
        axes[1].set_ylabel('Distance (pixel)')
        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_skimage_transformation_hough_line_peaks(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.transform import hough_line, hough_line_peaks
        from skimage.draw import line
        img = np.zeros((15, 15), dtype=np.bool_)
        rr, cc = line(0, 0, 14, 14)
        img[rr, cc] = 1
        rr, cc = line(0, 14, 14, 0)
        img[cc, rr] = 1
        hspace, angles, dists = hough_line(img)
        hspace, angles, dists = hough_line_peaks(hspace, angles, dists)

    """)
    selenium.run(cmd)

def test_skimage_transformation_integral_image(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.transform import integral_image
        arr = np.ones((5, 6), dtype=np.float)
        ii = integral_image(arr)
    """)
    selenium.run(cmd)

def test_skimage_transformation_rescale(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage import data
        from skimage.transform import rescale
        image = data.camera()
        rescale(image, 0.1).shape
    """)
    selenium.run(cmd)

def test_skimage_transformation_resize(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage import data
        from skimage.transform import resize
        image = data.camera()
        resize(image, (100, 100)).shape
    """)
    selenium.run(cmd)

def test_skimage_transformation_rotate(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage import data
        from skimage.transform import rotate
        image = data.camera()
        rotate(image, 2).shape
    """)
    selenium.run(cmd)

def test_skimage_transformation_warp(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage.transform import warp
        from skimage import data
        image = data.camera()
        from skimage.transform import SimilarityTransform
        tform = SimilarityTransform(translation=(0, -10))
        warped = warp(image, tform)
    """)
    selenium.run(cmd)

    
def test_skimage_transformation_warp_coords(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import data
        from skimage.transform import warp_coords
        from scipy.ndimage import map_coordinates
        def shift_up10_left20(xy):
            return xy - np.array([-20, 10])[None, :]
        image = data.coffee().astype(np.float32)
        coords = warp_coords(shift_up10_left20, image.shape)
        warped_image = map_coordinates(image, coords)
    """)
    selenium.run(cmd)

def test_skimage_transformation_warp_polar(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import data
        from skimage.transform import warp_polar
        image = data.checkerboard()
        warped = warp_polar(image)
    """)
    selenium.run(cmd)