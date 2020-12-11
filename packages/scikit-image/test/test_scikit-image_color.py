from textwrap import dedent

import pytest


def test_skimage_color_stains_combination_and_seperation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage import data
        from skimage.color import (separate_stains, combine_stains,
                                hdx_from_rgb, rgb_from_hdx)
        ihc = data.immunohistochemistry()
        ihc_hdx = separate_stains(ihc, hdx_from_rgb)
        ihc_rgb = combine_stains(ihc_hdx, rgb_from_hdx)
    """)
    selenium.run(cmd)

def test_skimage_color_convert_colorspace(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage import data
        from skimage.color import convert_colorspace
        img = data.astronaut()
        img_hsv = convert_colorspace(img, 'RGB', 'HSV')
    """)
    selenium.run(cmd)


def test_skimage_color_gray2rgb(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #gray2rgb
        from skimage import data
        from skimage import color
        from skimage import img_as_float

        grayscale_image = img_as_float(data.camera()[::2, ::2])
        image = color.gray2rgb(grayscale_image)
    """)
    selenium.run(cmd)


def test_skimage_color_hed2rgb(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #hed2rgb
        from skimage import data
        from skimage.color import rgb2hed, hed2rgb
        ihc = data.immunohistochemistry()
        ihc_hed = rgb2hed(ihc)
        ihc_rgb = hed2rgb(ihc_hed)
    """)
    selenium.run(cmd)


def test_skimage_color_lab2lch(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #lab2lch
        from skimage import data
        from skimage.color import rgb2lab, lab2lch
        img = data.astronaut()
        img_lab = rgb2lab(img)
        img_lch = lab2lch(img_lab)
    """)
    selenium.run(cmd)


def test_skimage_color_rgb2hsv(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #rgb2hsv
        from skimage import color
        from skimage import data
        img = data.astronaut()
        img_hsv = color.rgb2hsv(img)
    """)
    selenium.run(cmd)


def test_skimage_color_rgb2rgbcie(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #rgb2rgbcie
        from skimage import data
        from skimage.color import rgb2rgbcie
        img = data.astronaut()
        img_rgbcie = rgb2rgbcie(img)
    """)
    selenium.run(cmd)


def test_skimage_color_xyz2lab(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #xyz2lab
        from skimage import data
        from skimage.color import rgb2xyz, xyz2lab
        img = data.astronaut()
        img_xyz = rgb2xyz(img)
        img_lab = xyz2lab(img_xyz)
    """)
    selenium.run(cmd)



