from textwrap import dedent

import pytest

def test_scipy_interpolate_cubic(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.interpolate import interp1d
        x = np.linspace(0, 10, num=11, endpoint=True)
        y = np.cos(-x**2/9.0)
        f = interp1d(x, y)
        f2 = interp1d(x, y, kind='cubic')
    """)
    selenium.run(cmd)

def test_scipy_interpolate_multivariate(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        def func(x, y):
            return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
        points = np.random.rand(1000, 2)
        values = func(points[:,0], points[:,1])

        from scipy.interpolate import griddata
        grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
        grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    """)
    selenium.run(cmd)

def test_scipy_interpolate_splines(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import interpolate

        x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
        y = np.sin(x)
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.arange(0, 2*np.pi, np.pi/50)
        ynew = interpolate.splev(xnew, tck, der=0)
        yder = interpolate.splev(xnew, tck, der=1)


        def integ(x, tck, constant=-1):
            x = np.atleast_1d(x)
            out = np.zeros(x.shape, dtype=x.dtype)
            for n in range(len(out)):
                out[n] = interpolate.splint(0, x[n], tck)
            out += constant
            return out
        yint = integ(xnew, tck)

        interpolate.sproot(tck)

        t = np.arange(0, 1.1, .1)
        x = np.sin(2*np.pi*t)
        y = np.cos(2*np.pi*t)
        tck, u = interpolate.splprep([x, y], s=0)
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)

    """)
    selenium.run(cmd)

def test_scipy_interpolate_splines_mesh(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

        # setup data
        x = np.linspace(0, 10, 9)
        y = np.sin(x)
        xi = np.linspace(0, 10, 101)

        # use fitpack2 method
        ius = InterpolatedUnivariateSpline(x, y)
        yi = ius(xi)

        # use RBF method
        rbf = Rbf(x, y)
        fi = rbf(xi)

        #from matplotlib import cm

        # 2-d tests - setup scattered data
        x = np.random.rand(100)*4.0-2.0
        y = np.random.rand(100)*4.0-2.0
        z = x*np.exp(-x**2-y**2)
        edges = np.linspace(-2.0, 2.0, 101)
        centers = edges[:-1] + np.diff(edges[:2])[0] / 2.
        XI, YI = np.meshgrid(centers, centers)

        # use RBF
        rbf = Rbf(x, y, z, epsilon=2)
        ZI = rbf(XI, YI)

    """)
    selenium.run(cmd)
