from textwrap import dedent

import pytest


def test_scipy_import(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import signal
        from scipy import misc
        ascent = misc.ascent()
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                        [-10+0j, 0+ 0j, +10 +0j],
                        [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
        grad = signal.convolve2d(ascent, scharr, boundary='symm', mode='same')
        print(grad)
    """)
    selenium.run(cmd)

def test_scipy_linregress(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        import scipy.stats as sps
        # np.random.seed(12345678)
        # x = np.random.random(10)
        # y = 1.6*x + np.random.random(10)
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # print("slope: %f    intercept: %f" % (slope, intercept))    
    """)
    selenium.run(cmd)

def test_scipy_eigenvalues(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy import linalg
        import numpy as np
        a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
        print(linalg.eigvals(a, homogeneous_eigvals=True))
    """)
    selenium.run(cmd)


def test_scipy_fourier(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy import linalg
        import numpy as np
        np.set_printoptions(precision=2, suppress=True)  # for compact output
        m = linalg.dft(5)
        print(m)
        x = np.array([1, 2, 3, 0, 3])
        m @ x  # Compute the DFT of x
        from scipy.fft import fft
        print(fft(x))
    """)
    selenium.run(cmd)


def test_scipy_optimize(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.optimize import differential_evolution
        import numpy as np
        def ackley(x):
            arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
            arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
            return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
        bounds = [(-5, 5), (-5, 5)]
        result = differential_evolution(ackley, bounds)
        print(result.x, result.fun)
    """)
    selenium.run(cmd)

def test_scipy_rosen(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.optimize import rosen, differential_evolution
        bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
        result = differential_evolution(rosen, bounds)
        print(result.x, result.fun)
    """)
    selenium.run(cmd)


def test_scipy_lsq(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.optimize import least_squares
        def fun_rosenbrock(x):
            return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
        x0_rosenbrock = np.array([2, 2])
        res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
        print(res_1.x)
        print(res_1.cost)
        print(res_1.optimality)
    """)
    selenium.run(cmd)


def test_scipy_matrix_creation(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import linalg
        A = np.mat('[1 3 2; 1 4 5; 2 3 6]')
        T, Z = linalg.schur(A)
        T1, Z1 = linalg.schur(A, 'complex')
        T2, Z2 = linalg.rsf2csf(T, Z)
    """)
    selenium.run(cmd)


def test_scipy_linalg(selenium_standalone, request):
    selenium = selenium_standalone

    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))

    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        import scipy as sp
        import scipy.linalg
        from numpy.testing import assert_allclose

        N = 10
        X = np.random.RandomState(42).rand(N, N)

        X_inv = scipy.linalg.inv(X)

        res = X.dot(X_inv)

        assert_allclose(res, np.identity(N),
                        rtol=1e-07, atol=1e-9)
        """)

    selenium.run(cmd)


def test_brentq(selenium_standalone, request):
    selenium_standalone.load_package("scipy")

    if selenium_standalone.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(
            run=False, reason='chrome not supported'))

    selenium_standalone.run("from scipy.optimize import brentq")
    selenium_standalone.run("brentq(lambda x: x, -1, 1)")
