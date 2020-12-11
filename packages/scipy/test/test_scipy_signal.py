from textwrap import dedent

import pytest

def test_scipy_signal_b_splines(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import signal, misc
        image = misc.face(gray=True).astype(np.float32)
        derfilt = np.array([1.0, -2, 1.0], dtype=np.float32)
        ck = signal.cspline2d(image, 8.0)
        deriv = (signal.sepfir2d(ck, derfilt, [1]) +
                signal.sepfir2d(ck, [1], derfilt))

        laplacian = np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.float32)
        deriv2 = signal.convolve2d(ck,laplacian,mode='same',boundary='symm')
    """)
    selenium.run(cmd)

def test_scipy_signal_convolve(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy import signal
        import numpy as np
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        signal.convolve(x, h)
    """)
    selenium.run(cmd)

def test_scipy_signal_fftconvolve(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=True, reason='Possibly blocked on Dispatchable support in numpy. Need version 1.16'))
    selenium.load_package("scipy")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from scipy import signal, misc
        import matplotlib.pyplot as plt
        image = misc.face(gray=True)
        w = np.zeros((50, 50))
        w[0][0] = 1.0
        w[49][25] = 1.0
        image_new = signal.fftconvolve(image, w)
    """)
    selenium.run(cmd)

def test_scipy_signal_gaussian_sepfir2d(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from scipy import signal, misc
        import matplotlib.pyplot as plt
        image = misc.ascent()
        w = signal.gaussian(50, 10.0)
        image_new = signal.sepfir2d(image, w, w)
    """)
    selenium.run(cmd)