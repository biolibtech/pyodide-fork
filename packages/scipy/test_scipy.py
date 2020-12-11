from textwrap import dedent

import pytest


def test_scipy_import(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        import scipy.stats as sps
    """)
    selenium.run(cmd)
