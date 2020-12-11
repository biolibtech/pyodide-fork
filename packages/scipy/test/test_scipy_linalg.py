from textwrap import dedent

import pytest

def test_scipy_linalg_eigenvalues(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy import linalg
        import numpy as np
        a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
        linalg.eigvals(a, homogeneous_eigvals=True)
    """)
    selenium.run(cmd)


def test_scipy_linalg_basic_operations(selenium_standalone, request):
    selenium = selenium_standalone
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

def test_scipy_linalg_matrix_creation(selenium_standalone, request):
    selenium = selenium_standalone
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

def test_scipy_linalg_inverse(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import linalg
        A = np.array([[1,3,5],[2,5,1],[2,3,8]])
        linalg.inv(A)
        A.dot(linalg.inv(A)) #double check
    """)
    selenium.run(cmd)

def test_scipy_linalg_linear_system(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import linalg
        A = np.array([[1, 2], [3, 4]])
        b = np.array([[5], [6]])
        linalg.inv(A).dot(b)  # slow
        A.dot(linalg.inv(A).dot(b)) - b  # check
        np.linalg.solve(A, b) # fast
        A.dot(np.linalg.solve(A, b)) - b  # check
    """)
    selenium.run(cmd)

def test_scipy_linalg_least_squares(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import linalg
        c1, c2 = 5.0, 2.0
        i = np.r_[1:11]
        xi = 0.1*i
        yi = c1*np.exp(-xi) + c2*xi
        zi = yi + 0.05 * np.max(yi) * np.random.randn(len(yi))
        A = np.c_[np.exp(-xi)[:, np.newaxis], xi[:, np.newaxis]]
        c, resid, rank, sigma = linalg.lstsq(A, zi)
        xi2 = np.r_[0.1:1.0:100j]
        yi2 = c[0]*np.exp(-xi2) + c[1]*xi2
    """)
    selenium.run(cmd)

def test_scipy_linalg_singular_value_decomposition(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import linalg
        A = np.array([[1,2,3],[4,5,6]])
        M,N = A.shape
        U,s,Vh = linalg.svd(A)
        Sig = linalg.diagsvd(s,M,N)
        U, Vh = U, Vh
        U.dot(Sig.dot(Vh)) #check computation
    """)
    selenium.run(cmd)

def test_scipy_linalg_schurr_decomposition(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy import linalg
        import numpy as np
        A = np.mat('[1 3 2; 1 4 5; 2 3 6]')
        T, Z = linalg.schur(A)
        T1, Z1 = linalg.schur(A, 'complex')
        T2, Z2 = linalg.rsf2csf(T, Z)
    """)
    selenium.run(cmd)