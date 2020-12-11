from textwrap import dedent

import pytest


def test_scipy_optimize_differential_evolution(selenium_standalone, request):
    selenium = selenium_standalone
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
    """)
    selenium.run(cmd)


def test_brentq(selenium_standalone, request):
    selenium_standalone.load_package("scipy")

    selenium_standalone.run("from scipy.optimize import brentq")
    selenium_standalone.run("brentq(lambda x: x, -1, 1)")
    

def test_scipy_optimize_rosen_der_and_hess(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.optimize import minimize, rosen
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])


        def rosen_der(x):
            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            der = np.zeros_like(x)
            der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
            der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
            der[-1] = 200*(x[-1]-x[-2]**2)
            return der
        
        res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
               options={'disp': True})

        def rosen_hess(x):
            x = np.asarray(x)
            H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
            diagonal = np.zeros_like(x)
            diagonal[0] = 1200*x[0]**2-400*x[1]+2
            diagonal[-1] = 200
            diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
            H = H + np.diag(diagonal)
            return H

        res = minimize(rosen, x0, method='Newton-CG',
               jac=rosen_der, hess=rosen_hess,
               options={'xtol': 1e-8, 'disp': True})
    """)
    selenium.run(cmd)

def test_scipy_optimize_constrains_and_bounds(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.optimize import Bounds, minimize, rosen
        import numpy as np
        x0 = np.array([0.5, 0])

        def rosen_der(x):
            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            der = np.zeros_like(x)
            der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
            der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
            der[-1] = 200*(x[-1]-x[-2]**2)
            return der

        def rosen_hess(x):
            x = np.asarray(x)
            H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
            diagonal = np.zeros_like(x)
            diagonal[0] = 1200*x[0]**2-400*x[1]+2
            diagonal[-1] = 200
            diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
            H = H + np.diag(diagonal)
            return H

        bounds = Bounds([0, -0.5], [1.0, 2.0])

        #Linear constraint
        from scipy.optimize import LinearConstraint
        linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
        

        # Non linear constraint
        def cons_f(x):
            return [x[0]**2 + x[1], x[0]**2 - x[1]]
        def cons_J(x):
            return [[2*x[0], 1], [2*x[0], -1]]
        def cons_H(x, v):
            return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])
        from scipy.optimize import NonlinearConstraint
        nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

        # With Sparse Matrix
        from scipy.sparse import csc_matrix
        def cons_H_sparse(x, v):
            return v[0]*csc_matrix([[2, 0], [0, 0]]) + v[1]*csc_matrix([[2, 0], [0, 0]])
        nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1,
                                                jac=cons_J, hess=cons_H_sparse)

        res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
               constraints=[linear_constraint, nonlinear_constraint],
               options={'verbose': 1}, bounds=bounds)

    """)
    selenium.run(cmd)

def test_scipy_optimize_minize_scalar(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.optimize import minimize_scalar
        import numpy as np
        f = lambda x: (x - 2) * (x + 1)**2
        res = minimize_scalar(f, method='brent')

        # Bounded
        from scipy.special import j1
        res = minimize_scalar(j1, bounds=(4, 7), method='bounded')
    """)
    selenium.run(cmd)

def test_scipy_optimize_root_finding(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.optimize import root
        def func(x):
            return x + 2 * np.cos(x)
        sol = root(func, 0.3)

        def func2(x):
            f = [x[0] * np.cos(x[1]) - 4,
                x[1]*x[0] - x[1] - 5]
            df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])],
                        [x[1], x[0] - 1]])
            return f, df
        sol = root(func2, [1, 1], jac=True, method='lm')
    """)
    selenium.run(cmd)

def test_scipy_optimize_root_finding_krylov_large(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.optimize import root
        from numpy import cosh, zeros_like, mgrid, zeros

        # parameters
        nx, ny = 75, 75
        hx, hy = 1./(nx-1), 1./(ny-1)

        P_left, P_right = 0, 0
        P_top, P_bottom = 1, 0

        def residual(P):
            d2x = zeros_like(P)
            d2y = zeros_like(P)

            d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx
            d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
            d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

            d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
            d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
            d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

            return d2x + d2y + 5*cosh(P).mean()**2

        # solve
        guess = zeros((nx, ny), float)
        sol = root(residual, guess, method='krylov', options={'disp': True})
        #sol = root(residual, guess, method='broyden2', options={'disp': True, 'max_rank': 50})
        #sol = root(residual, guess, method='anderson', options={'disp': True, 'M': 10})
        abs(residual(sol.x)).max()
    """)
    selenium.run(cmd)

def test_scipy_optimize_root_finding(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.optimize import linprog
        c = np.array([-29.0, -45.0, 0.0, 0.0])
        A_ub = np.array([[1.0, -1.0, -3.0, 0.0],
                        [-2.0, 3.0, 7.0, -3.0]])
        b_ub = np.array([5.0, -10.0])
        A_eq = np.array([[2.0, 8.0, 1.0, 0.0],
                        [4.0, 4.0, 0.0, 1.0]])
        b_eq = np.array([60.0, 60.0])
        x0_bounds = (0, None)
        x1_bounds = (0, 5.0)
        x2_bounds = (-np.inf, 0.5)  # +/- np.inf can be used instead of None
        x3_bounds = (-3.0, None)
        bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    """)
    selenium.run(cmd)


def test_scipy_optimize_least_squares(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.optimize import least_squares
        def fun_rosenbrock(x):
            return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
        x0_rosenbrock = np.array([2, 2])
        res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
    """)
    selenium.run(cmd)