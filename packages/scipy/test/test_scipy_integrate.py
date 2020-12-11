from textwrap import dedent

import pytest

def test_scipy_integrate_quad(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=False, reason='ctypes is not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import scipy.integrate as integrate
        import scipy.special as special
        result = integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)
        from numpy import sqrt, sin, cos, pi
        I = sqrt(2/pi)*(18.0/27*sqrt(2)*cos(4.5) - 4.0/27*sqrt(2)*sin(4.5) +
                        sqrt(2*pi) * special.fresnel(3/sqrt(pi))[0])

    """)
    selenium.run(cmd)


def test_scipy_integrate_multiple_integration(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=False, reason='ctypes is not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.integrate import quad, dblquad
        import numpy as np
        def I(n):
            return dblquad(lambda t, x: np.exp(-x*t)/t**n, 0, np.inf, lambda x: 1, lambda x: np.inf)
    """)
    selenium.run(cmd)


def test_scipy_integrate_ode(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.integrate import solve_ivp
        from scipy.special import gamma, airy
        import numpy as np
        y1_0 = +1 / 3**(2/3) / gamma(2/3)
        y0_0 = -1 / 3**(1/3) / gamma(1/3)
        y0 = [y0_0, y1_0]
        def func(t, y):
            return [t*y[1],y[0]]
        t_span = [0, 4]
        sol1 = solve_ivp(func, t_span, y0)
    """)
    selenium.run(cmd)


def test_scipy_integrate_ode_jacobian(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=False, reason='_odepack fortran is broken'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.integrate import odeint
        def G(u, v, f, k):
            return f * (1 - u) - u*v**2

        def H(u, v, f, k):
            return -(f + k) * v + u*v**2

        def grayscott1d(y, t, f, k, Du, Dv, dx):
            # The vectors u and v are interleaved in y.  We define
            # views of u and v by slicing y.
            u = y[::2]
            v = y[1::2]

            # dydt is the return value of this function.
            dydt = np.empty_like(y)

            # Just like u and v are views of the interleaved vectors
            # in y, dudt and dvdt are views of the interleaved output
            # vectors in dydt.
            dudt = dydt[::2]
            dvdt = dydt[1::2]

            # Compute du/dt and dv/dt.  The end points and the interior points
            # are handled separately.
            dudt[0]    = G(u[0],    v[0],    f, k) + Du * (-2.0*u[0] + 2.0*u[1]) / dx**2
            dudt[1:-1] = G(u[1:-1], v[1:-1], f, k) + Du * np.diff(u,2) / dx**2
            dudt[-1]   = G(u[-1],   v[-1],   f, k) + Du * (- 2.0*u[-1] + 2.0*u[-2]) / dx**2
            dvdt[0]    = H(u[0],    v[0],    f, k) + Dv * (-2.0*v[0] + 2.0*v[1]) / dx**2
            dvdt[1:-1] = H(u[1:-1], v[1:-1], f, k) + Dv * np.diff(v,2) / dx**2
            dvdt[-1]   = H(u[-1],   v[-1],   f, k) + Dv * (-2.0*v[-1] + 2.0*v[-2]) / dx**2

            return dydt

        y0 = np.random.randn(5000)
        t = np.linspace(0, 50, 11)
        f = 0.024
        k = 0.055
        Du = 0.01
        Dv = 0.005
        dx = 0.025

        odeint(grayscott1d, y0, t, args=(f, k, Du, Dv, dx))
    """)
    selenium.run(cmd)