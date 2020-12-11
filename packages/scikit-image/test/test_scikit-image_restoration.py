from textwrap import dedent
import pytest


def test_skimage_restoration_denoising_function(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        #optical_flow_tvl1
        from skimage import color, data
        from skimage.restoration import denoise_wavelet, calibrate_denoiser
        import numpy as np
        img = color.rgb2gray(data.coffee()[:50, :50])
        noisy = img + 0.5 * img.std() * np.random.randn(*img.shape)
        parameters = {'sigma': np.arange(0.1, 0.4, 0.02)}
        denoising_function = calibrate_denoiser(noisy, denoise_wavelet,
                                                denoise_parameters=parameters)
        denoised_img = denoising_function(img)
    """)
    selenium.run(cmd)

def test_skimage_restoration_cycle_spin(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        import skimage.data
        from skimage import img_as_float
        from skimage.restoration import denoise_wavelet, cycle_spin
        img = img_as_float(skimage.data.camera())
        sigma = 0.1
        img = img + sigma * np.random.standard_normal(img.shape)
        denoised = cycle_spin(img, func=denoise_wavelet,
                            max_shifts=3) 
    """)
    selenium.run(cmd)

def test_skimage_restoration_denoise_bilateral(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import data, img_as_float
        from skimage.restoration import denoise_bilateral
        astro = img_as_float(data.coffee())
        astro = astro[220:300, 220:320]
        noisy = astro + 0.6 * astro.std() * np.random.random(astro.shape)
        noisy = np.clip(noisy, 0, 1)
        denoised = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
                                    multichannel=True)
    """)
    selenium.run(cmd)

def test_skimage_restoration_denoise_nl_means(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.restoration import denoise_nl_means
        a = np.zeros((40, 40))
        a[10:-10, 10:-10] = 1.
        a += 0.3 * np.random.randn(*a.shape)
        denoised_a = denoise_nl_means(a, 7, 5, 0.1)
    """)
    selenium.run(cmd)

def test_skimage_restoration_denoise_tv_chambolle(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage.restoration import denoise_tv_chambolle
        import numpy as np
        x, y, z = np.ogrid[0:20, 0:20, 0:20]
        mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
        mask = mask.astype(np.float)
        mask += 0.2*np.random.randn(*mask.shape)
        res = denoise_tv_chambolle(mask, weight=100)
    """)
    selenium.run(cmd)

def test_skimage_restoration_denoise_wavelet(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage import color, data
        from skimage import img_as_float
        from skimage.restoration import denoise_wavelet
        import numpy as np
        img = img_as_float(data.coffee())
        img = color.rgb2gray(img)
        img += 0.1 * np.random.randn(*img.shape)
        img = np.clip(img, 0, 1)
        denoised_img = denoise_wavelet(img, sigma=0.1, rescale_sigma=True)

    """)
    selenium.run(cmd)

def test_skimage_restoration_estimate_sigma(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import skimage.data
        from skimage import img_as_float
        from skimage.restoration import estimate_sigma
        import numpy as np
        img = img_as_float(skimage.data.camera())
        sigma = 0.1
        img = img + sigma * np.random.standard_normal(img.shape)
        sigma_hat = estimate_sigma(img, multichannel=False)
    """)
    selenium.run(cmd)

def test_skimage_restoration_inpaint_biharmonic(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        from skimage.restoration import inpaint_biharmonic
        import numpy as np
        img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
        mask = np.zeros_like(img)
        mask[2, 2:] = 1
        mask[1, 3:] = 1
        mask[0, 4:] = 1
        out = inpaint_biharmonic(img, mask)
    """)
    selenium.run(cmd)

def test_skimage_restoration_richardson_lucy(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import color, data, restoration
        camera = color.rgb2gray(data.camera())
        from scipy.signal import convolve2d
        psf = np.ones((5, 5)) / 25
        camera = convolve2d(camera, psf, 'same')
        camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
        deconvolved = restoration.richardson_lucy(camera, psf, 5)

    """)
    selenium.run(cmd)

def test_skimage_restoration_unsupervised_wiener(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage import color, data, restoration
        img = color.rgb2gray(data.coffee())
        from scipy.signal import convolve2d
        psf = np.ones((5, 5)) / 25
        img = convolve2d(img, psf, 'same')
        img += 0.1 * img.std() * np.random.standard_normal(img.shape)
        deconvolved_img = restoration.unsupervised_wiener(img, psf)

    """)
    selenium.run(cmd)

def test_skimage_restoration_unwrap_phase(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-image")
    cmd = dedent(r"""
        import numpy as np
        from skimage.restoration import unwrap_phase
        c0, c1 = np.ogrid[-1:1:128j, -1:1:128j]
        image = 12 * np.pi * np.exp(-(c0**2 + c1**2))
        image_wrapped = np.angle(np.exp(1j * image))
        image_unwrapped = unwrap_phase(image_wrapped)
        np.std(image_unwrapped - image) < 1e-6   # A constant offset is normal
    """)
    selenium.run(cmd)
