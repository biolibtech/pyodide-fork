from textwrap import dedent

import pytest

def test_scipy_stats_norm_functions(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.stats import norm
        norm.cdf([-1., 0, 1])
        norm.mean(), norm.std(), norm.var()
        norm.stats(moments="mv")
        norm.ppf(0.5)
        norm.rvs(size=3)
        norm.stats(loc=3, scale=4, moments="mv")
    """)
    selenium.run(cmd)

def test_scipy_stats_scaling(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.stats import uniform
        uniform.cdf([0, 1, 2, 3, 4, 5], loc=1, scale=4)
    """)
    selenium.run(cmd)

def test_scipy_stats_gamma_shape(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy.stats import gamma
        gamma.numargs
        gamma.shapes
    """)
    selenium.run(cmd)

def test_scipy_stats_broadcasting(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy import stats
        stats.t.isf([0.1, 0.05, 0.01], [10, 11, 12])
    """)
    selenium.run(cmd)

def test_scipy_stats_building_distributions(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=False, reason='ctypes is not supported'))
    selenium.load_package("scipy")
    cmd = dedent(r"""
        from scipy import stats
        import numpy as np
        class deterministic_gen(stats.rv_continuous):
            def _cdf(self, x):
                return np.where(x < 0, 0., 1.)
            def _stats(self):
                return 0., 0., 0., 0.
        deterministic = deterministic_gen(name="deterministic")
        deterministic.cdf(np.arange(-3, 3, 0.5))
        
        from scipy.integrate import quad
        quad(deterministic.pdf, -1e-1, 1e-1)
    """)
    selenium.run(cmd)

def test_scipy_stats_descriptive_stats(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import stats
        np.random.seed(282629734)
        x = stats.t.rvs(10, size=1000)
        x.min()  # equivalent to np.min(x)
        x.max()  # equivalent to np.max(x)
        x.mean() # equivalent to np.mean(x)
        x.var()  # equivalent to np.var(x))
        m, v, s, k = stats.t.stats(10, moments='mvsk')
        stats.describe(x)
    """)
    selenium.run(cmd)

def test_scipy_stats_t_and_ks_test(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import stats
        np.random.seed(282629734)
        x = stats.t.rvs(10, size=1000)
        m, v, s, k = stats.t.stats(10, moments='mvsk')
        n, (smin, smax), sm, sv, ss, sk = stats.describe(x)
        stats.ttest_1samp(x, m)
        (sm-m)/np.sqrt(sv/float(n))
        stats.kstest(x, 't', (10,))
        d, pval = stats.kstest((x-x.mean())/x.std(), 'norm')
    """)
    selenium.run(cmd)

def test_scipy_stats_broadcasting(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy import stats
        crit01, crit05, crit10 = stats.t.ppf([1-0.01, 1-0.05, 1-0.10], 10)
        freq01 = np.sum(x>crit01) / float(n) * 100
        freq05 = np.sum(x>crit05) / float(n) * 100
        freq10 = np.sum(x>crit10) / float(n) * 100
        freq05l = np.sum(stats.t.rvs(10, size=10000) > crit05) / 10000.0 * 100
    """)
    selenium.run(cmd)