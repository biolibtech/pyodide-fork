from textwrap import dedent

import pytest

def test_scipy_fft_simple_fft(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.fft import fft, ifft
        import numpy as np
        x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
        y = fft(x)
        yinv = ifft(y)
    """)
    selenium.run(cmd)

def test_scipy_fft_simple_fft(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.fft import fft
        import numpy as np
        # Number of sample points
        N = 600
        # sample spacing
        T = 1.0 / 800.0
        x = np.linspace(0.0, N*T, N)
        y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
        yf = fft(y)
        from scipy.signal import blackman
        w = blackman(N)
        ywf = fft(y*w)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    """)
    selenium.run(cmd)

def test_scipy_fft_misc_fft_functions(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.fft import fft, fftfreq, fftshift
        import numpy as np
        # number of signal points
        N = 400
        # sample spacing
        T = 1.0 / 800.0
        x = np.linspace(0.0, N*T, N)
        y = np.exp(50.0 * 1.j * 2.0*np.pi*x) + 0.5*np.exp(-80.0 * 1.j * 2.0*np.pi*x)
        yf = fft(y)
        xf = fftfreq(N, T)
        xf = fftshift(xf)
        yplot = fftshift(yf)

    """)
    selenium.run(cmd)

def test_scipy_fft_n_d_fft(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.fft import ifftn
        N = 30
        xf = np.zeros((N,N))
        xf[0, 5] = 1
        xf[0, N-5] = 1
        Z = ifftn(xf)
        xf = np.zeros((N, N))
        xf[5, 0] = 1
        xf[N-5, 0] = 1
        Z = ifftn(xf)
        xf = np.zeros((N, N))
        xf[5, 10] = 1
        xf[N-5, N-10] = 1
        Z = ifftn(xf)
    """)
    selenium.run(cmd)

def test_scipy_fft_discrete_cosine_transform(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=True, reason='Blocked on Dispatchable support in numpy. Need version 1.16'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.fft import dct, idct
        import numpy as np
        N = 100
        t = np.linspace(0,20,N)
        x = np.exp(-t/3)*np.cos(2*t)
        y = dct(x, norm='ortho')
        window1 = np.zeros(N)
        window1[:20] = 1
        yr = idct(y*window1, norm='ortho')
        sum(abs(x-yr)**2) / sum(abs(x)**2)

        window1 = np.zeros(N)
        window1[:15] = 1
        yr = idct(y*window1, norm='ortho')
        sum(abs(x-yr)**2) / sum(abs(x)**2)
    """)
    selenium.run(cmd)