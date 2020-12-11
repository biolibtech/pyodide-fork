from textwrap import dedent
import pytest


def test_matplotlib(selenium_standalone, request):
    selenium = selenium_standalone
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(run=False, reason='chrome not supported'))
    selenium.load_package("matplotlib")
    selenium.run("from matplotlib import pyplot as plt")
    selenium.run("plt.figure()")
    selenium.run("plt.plot([1,2,3])")
    selenium.run("plt.show()")


def test_svg(selenium, request):
    selenium.load_package("matplotlib")
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(run=False, reason='chrome not supported'))
    selenium.run("from matplotlib import pyplot as plt")
    selenium.run("plt.figure()")
    selenium.run("x = plt.plot([1,2,3])")
    selenium.run("import io")
    selenium.run("fd = io.BytesIO()")
    selenium.run("plt.savefig(fd, format='svg')")
    content = selenium.run("fd.getvalue().decode('utf8')")
    assert len(content) == 15753
    assert content.startswith("<?xml")


def test_pdf(selenium, request):
    selenium.load_package("matplotlib")
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(run=False, reason='chrome not supported'))
    selenium.run("from matplotlib import pyplot as plt")
    selenium.run("plt.figure()")
    selenium.run("x = plt.plot([1,2,3])")
    selenium.run("import io")
    selenium.run("fd = io.BytesIO()")
    selenium.run("plt.savefig(fd, format='pdf')")

