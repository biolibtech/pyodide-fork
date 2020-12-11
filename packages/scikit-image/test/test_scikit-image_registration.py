from textwrap import dedent
import pytest


def test_skimage_registration_optical_flow_tvl1(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #optical_flow_tvl1
        from skimage.color import rgb2gray
        from skimage.data import stereo_motorcycle
        from skimage.registration import optical_flow_tvl1
        image0, image1, disp = stereo_motorcycle()
        # --- Convert the images to gray level: color is not supported.
        image0 = rgb2gray(image0)
        image1 = rgb2gray(image1)
        flow = optical_flow_tvl1(image1, image0)
    """)
    selenium.run(cmd)