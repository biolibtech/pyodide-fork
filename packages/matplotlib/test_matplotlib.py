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


def test_stacked_bar(selenium, request):
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(run=False, reason='chrome not supported'))
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt


        labels = ['G1', 'G2', 'G3', 'G4', 'G5']
        men_means = [20, 35, 30, 35, 27]
        women_means = [25, 32, 34, 20, 25]
        men_std = [2, 3, 4, 1, 2]
        women_std = [3, 5, 2, 3, 3]
        width = 0.35       # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()

        ax.bar(labels, men_means, width, yerr=men_std, label='Men')
        ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
            label='Women')

        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.legend()

        plt.show()
    """)
    selenium.run(cmd)


def test_CSD(selenium, request):
    if selenium.browser == 'chrome':
        request.applymarker(pytest.mark.xfail(run=False, reason='chrome not supported'))
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1)
        # make a little extra space between the subplots
        fig.subplots_adjust(hspace=0.5)

        dt = 0.01
        t = np.arange(0, 30, dt)

        # Fixing random state for reproducibility
        np.random.seed(19680801)


        nse1 = np.random.randn(len(t))                 # white noise 1
        nse2 = np.random.randn(len(t))                 # white noise 2
        r = np.exp(-t / 0.05)

        cnse1 = np.convolve(nse1, r, mode='same') * dt   # colored noise 1
        cnse2 = np.convolve(nse2, r, mode='same') * dt   # colored noise 2

        # two signals with a coherent part and a random part
        s1 = 0.01 * np.sin(2 * np.pi * 10 * t) + cnse1
        s2 = 0.01 * np.sin(2 * np.pi * 10 * t) + cnse2

        ax1.plot(t, s1, t, s2)
        ax1.set_xlim(0, 5)
        ax1.set_xlabel('time')
        ax1.set_ylabel('s1 and s2')
        ax1.grid(True)

        cxy, f = ax2.csd(s1, s2, 256, 1. / dt)
        ax2.set_ylabel('CSD (db)')
        plt.show()
    """)
    selenium.run(cmd)
