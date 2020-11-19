import pytest
from textwrap import dedent



def test_chemfp_import(selenium_standalone, request):
    selenium = selenium_standalone
    print(selenium.load_package("chemfp"))
    cmd = dedent(r"""
       import chemfp
       """)

    selenium.run(cmd)

