from textwrap import dedent
import pytest

def test_matplotlib_bars_lines_markers_plotting_categorical_variables(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt

        data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}
        names = list(data.keys())
        values = list(data.values())

        fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        axs[0].bar(names, values)
        axs[1].scatter(names, values)
        axs[2].plot(names, values)
        fig.suptitle('Categorical Plotting')
        cat = ["bored", "happy", "bored", "bored", "happy", "bored"]
        dog = ["happy", "happy", "happy", "happy", "bored", "bored"]
        activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

        fig, ax = plt.subplots()
        ax.plot(activity, dog, label="dog")
        ax.plot(activity, cat, label="cat")
        ax.legend()

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_curve_with_error_band(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    selenium.load_package("scipy")
    cmd = dedent(r"""
        import numpy as np
        from scipy.interpolate import splprep, splev

        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        N = 400
        t = np.linspace(0, 2 * np.pi, N)
        r = 0.5 + np.cos(t)
        x, y = r * np.cos(t), r * np.sin(t)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()
        err = 0.05 * np.sin(2 * t) ** 2 + 0.04 + 0.02 * np.cos(9 * t + 2)

        # calculate normals via derivatives of splines
        tck, u = splprep([x, y], s=0)
        dx, dy = splev(u, tck, der=1)
        l = np.hypot(dx, dy)
        nx = dy / l
        ny = -dx / l

        # end points of errors
        xp = x + nx * err
        yp = y + ny * err
        xn = x - nx * err
        yn = y - ny * err

        vertices = np.block([[xp, xn[::-1]],
                            [yp, yn[::-1]]]).T
        codes = Path.LINETO * np.ones(len(vertices), dtype=Path.code_type)
        codes[0] = codes[len(xp)] = Path.MOVETO
        path = Path(vertices, codes)

        patch = PathPatch(path, facecolor='C0', edgecolor='none', alpha=0.3)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.add_patch(patch)
        plt.show()

    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_errorbars_subsampling(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        # example data
        x = np.arange(0.1, 4, 0.1)
        y1 = np.exp(-1.0 * x)
        y2 = np.exp(-0.5 * x)

        # example variable error bar values
        y1err = 0.1 + 0.1 * np.sqrt(x)
        y2err = 0.1 + 0.1 * np.sqrt(x/2)


        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                            figsize=(12, 6))

        ax0.set_title('all errorbars')
        ax0.errorbar(x, y1, yerr=y1err)
        ax0.errorbar(x, y2, yerr=y2err)

        ax1.set_title('only every 6th errorbar')
        ax1.errorbar(x, y1, yerr=y1err, errorevery=6)
        ax1.errorbar(x, y2, yerr=y2err, errorevery=6)

        ax2.set_title('second series shifted by 3')
        ax2.errorbar(x, y1, yerr=y1err, errorevery=(0, 6))
        ax2.errorbar(x, y2, yerr=y2err, errorevery=(3, 6))

        fig.suptitle('Errorbar subsampling')
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_eventplot(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib
        matplotlib.rcParams['font.size'] = 8.0

        # Fixing random state for reproducibility
        np.random.seed(19680801)


        # create random data
        data1 = np.random.random([6, 50])

        # set different colors for each set of positions
        colors1 = ['C{}'.format(i) for i in range(6)]

        # set different line properties for each set of positions
        # note that some overlap
        lineoffsets1 = [-15, -3, 1, 1.5, 6, 10]
        linelengths1 = [5, 2, 1, 1, 3, 1.5]

        fig, axs = plt.subplots(2, 2)

        # create a horizontal plot
        axs[0, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                            linelengths=linelengths1)

        # create a vertical plot
        axs[1, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                            linelengths=linelengths1, orientation='vertical')

        # create another set of random data.
        # the gamma distribution is only used for aesthetic purposes
        data2 = np.random.gamma(4, size=[60, 50])

        # use individual values for the parameters this time
        # these values will be used for all data sets (except lineoffsets2, which
        # sets the increment between each data set in this usage)
        colors2 = 'black'
        lineoffsets2 = 1
        linelengths2 = 1

        # create a horizontal plot
        axs[0, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                            linelengths=linelengths2)


        # create a vertical plot
        axs[1, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                            linelengths=linelengths2, orientation='vertical')

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_hatch_filled_hist(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import itertools
        from collections import OrderedDict
        from functools import partial

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from cycler import cycler


        def filled_hist(ax, edges, values, bottoms=None, orientation='v',
                        **kwargs):
            print(orientation)
            if orientation not in 'hv':
                raise ValueError("orientation must be in {{'h', 'v'}} "
                                "not {o}".format(o=orientation))

            kwargs.setdefault('step', 'post')
            edges = np.asarray(edges)
            values = np.asarray(values)
            if len(edges) - 1 != len(values):
                raise ValueError('Must provide one more bin edge than value not: '
                                'len(edges): {lb} len(values): {lv}'.format(
                                    lb=len(edges), lv=len(values)))

            if bottoms is None:
                bottoms = 0
            bottoms = np.broadcast_to(bottoms, values.shape)

            values = np.append(values, values[-1])
            bottoms = np.append(bottoms, bottoms[-1])
            if orientation == 'h':
                return ax.fill_betweenx(edges, values, bottoms,
                                        **kwargs)
            elif orientation == 'v':
                return ax.fill_between(edges, values, bottoms,
                                    **kwargs)
            else:
                raise AssertionError("you should never be here")


        def stack_hist(ax, stacked_data, sty_cycle, bottoms=None,
                    hist_func=None, labels=None,
                    plot_func=None, plot_kwargs=None):
            # deal with default binning function
            if hist_func is None:
                hist_func = np.histogram

            # deal with default plotting function
            if plot_func is None:
                plot_func = filled_hist

            # deal with default
            if plot_kwargs is None:
                plot_kwargs = {}
            print(plot_kwargs)
            try:
                l_keys = stacked_data.keys()
                label_data = True
                if labels is None:
                    labels = l_keys

            except AttributeError:
                label_data = False
                if labels is None:
                    labels = itertools.repeat(None)

            if label_data:
                loop_iter = enumerate((stacked_data[lab], lab, s)
                                    for lab, s in zip(labels, sty_cycle))
            else:
                loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))

            arts = {}
            for j, (data, label, sty) in loop_iter:
                if label is None:
                    label = 'dflt set {n}'.format(n=j)
                label = sty.pop('label', label)
                vals, edges = hist_func(data)
                if bottoms is None:
                    bottoms = np.zeros_like(vals)
                top = bottoms + vals
                print(sty)
                sty.update(plot_kwargs)
                print(sty)
                ret = plot_func(ax, edges, top, bottoms=bottoms,
                                label=label, **sty)
                bottoms = top
                arts[label] = ret
            ax.legend(fontsize=10)
            return arts


        # set up histogram function to fixed bins
        edges = np.linspace(-3, 3, 20, endpoint=True)
        hist_func = partial(np.histogram, bins=edges)

        # set up style cycles
        color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:4])
        label_cycle = cycler(label=['set {n}'.format(n=n) for n in range(4)])
        hatch_cycle = cycler(hatch=['/', '*', '+', '|'])

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        stack_data = np.random.randn(4, 12250)
        dict_data = OrderedDict(zip((c['label'] for c in label_cycle), stack_data))
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_fill_betweenx(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np


        y = np.arange(0.0, 2, 0.01)
        x1 = np.sin(2 * np.pi * y)
        x2 = 1.2 * np.sin(4 * np.pi * y)

        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize=(6, 6))

        ax1.fill_betweenx(y, 0, x1)
        ax1.set_title('between (x1, 0)')

        ax2.fill_betweenx(y, x1, 1)
        ax2.set_title('between (x1, 1)')
        ax2.set_xlabel('x')

        ax3.fill_betweenx(y, x1, x2)
        ax3.set_title('between (x1, x2)')

        # now fill between x1 and x2 where a logical condition is met.  Note
        # this is different than calling
        #   fill_between(y[where], x1[where], x2[where])
        # because of edge effects over multiple contiguous regions.

        fig, [ax, ax1] = plt.subplots(1, 2, sharey=True, figsize=(6, 6))
        ax.plot(x1, y, x2, y, color='black')
        ax.fill_betweenx(y, x1, x2, where=x2 >= x1, facecolor='green')
        ax.fill_betweenx(y, x1, x2, where=x2 <= x1, facecolor='red')
        ax.set_title('fill_betweenx where')

        # Test support for masked arrays.
        x2 = np.ma.masked_greater(x2, 1.0)
        ax1.plot(x1, y, x2, y, color='black')
        ax1.fill_betweenx(y, x1, x2, where=x2 >= x1, facecolor='green')
        ax1.fill_betweenx(y, x1, x2, where=x2 <= x1, facecolor='red')
        ax1.set_title('regions with x2 > 1 are masked')

        # This example illustrates a problem; because of the data
        # gridding, there are undesired unfilled triangles at the crossover
        # points.  A brute-force solution would be to interpolate all
        # arrays to a very fine grid before plotting.

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_bar_charts_with_gradients(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np

        np.random.seed(19680801)

        def gradient_image(ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):
            phi = direction * np.pi / 2
            v = np.array([np.cos(phi), np.sin(phi)])
            X = np.array([[v @ [1, 0], v @ [1, 1]],
                        [v @ [0, 0], v @ [0, 1]]])
            a, b = cmap_range
            X = a + (b - a) / X.max() * X
            im = ax.imshow(X, extent=extent, interpolation='bicubic',
                        vmin=0, vmax=1, **kwargs)
            return im


        def gradient_bar(ax, x, y, width=0.5, bottom=0):
            for left, top in zip(x, y):
                right = left + width
                gradient_image(ax, extent=(left, right, bottom, top),
                            cmap=plt.cm.Blues_r, cmap_range=(0, 0.8))


        xmin, xmax = xlim = 0, 10
        ymin, ymax = ylim = 0, 1

        fig, ax = plt.subplots()
        ax.set(xlim=xlim, ylim=ylim, autoscale_on=False)

        # background image
        gradient_image(ax, direction=0, extent=(0, 1, 0, 1), transform=ax.transAxes,
                    cmap=plt.cm.Oranges, cmap_range=(0.1, 0.6))

        N = 10
        x = np.arange(N) + 0.15
        y = np.random.rand(N)
        gradient_bar(ax, x, y, width=0.7)
        ax.set_aspect('auto')
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_join_and_cap_styles(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt


        def plot_angle(ax, x, y, angle, style):
            phi = np.radians(angle)
            xx = [x + .5, x, x + .5*np.cos(phi)]
            yy = [y, y, y + .5*np.sin(phi)]
            ax.plot(xx, yy, lw=12, color='tab:blue', solid_joinstyle=style)
            ax.plot(xx, yy, lw=1, color='black')
            ax.plot(xx[1], yy[1], 'o', color='tab:red', markersize=3)


        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Join style')

        for x, style in enumerate(['miter', 'round', 'bevel']):
            ax.text(x, 5, style)
            for y, angle in enumerate([20, 45, 60, 90, 120]):
                plot_angle(ax, x, y, angle, style)
                if x == 0:
                    ax.text(-1.3, y, f'{angle} degrees')
        ax.text(1, 4.7, '(default)')

        ax.set_xlim(-1.5, 2.75)
        ax.set_ylim(-.5, 5.5)
        ax.set_axis_off()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.set_title('Cap style')

        for x, style in enumerate(['butt', 'round', 'projecting']):
            ax.text(x+0.25, 1, style, ha='center')
            xx = [x, x+0.5]
            yy = [0, 0]
            ax.plot(xx, yy, lw=12, color='tab:blue', solid_capstyle=style)
            ax.plot(xx, yy, lw=1, color='black')
            ax.plot(xx, yy, 'o', color='tab:red', markersize=3)
        ax.text(2.25, 0.7, '(default)', ha='center')

        ax.set_ylim(-.5, 1.5)
        ax.set_axis_off()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_dashed_lines_custom(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.linspace(0, 10, 500)
        y = np.sin(x)

        fig, ax = plt.subplots()

        # Using set_dashes() to modify dashing of an existing line
        line1, = ax.plot(x, y, label='Using set_dashes()')
        line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

        # Using plot(..., dashes=...) to set the dashing when creating a line
        line2, = ax.plot(x, y - 0.2, dashes=[6, 2], label='Using the dashes parameter')

        ax.legend()
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_masked_NaN_plotting(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np


        x = np.linspace(-np.pi/2, np.pi/2, 31)
        y = np.cos(x)**3

        # 1) remove points where y > 0.7
        x2 = x[y <= 0.7]
        y2 = y[y <= 0.7]

        # 2) mask points where y > 0.7
        y3 = np.ma.masked_where(y > 0.7, y)

        # 3) set to NaN where y > 0.7
        y4 = y.copy()
        y4[y3 > 0.7] = np.nan

        plt.plot(x*0.1, y, 'o-', color='lightgrey', label='No mask')
        plt.plot(x2*0.4, y2, 'o-', label='Points removed')
        plt.plot(x*0.7, y3, 'o-', label='Masked values')
        plt.plot(x*1.0, y4, 'o-', label='NaN values')
        plt.legend()
        plt.title('Masked and NaN data')
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_multicolored_lines(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap, BoundaryNorm

        x = np.linspace(0, 3 * np.pi, 500)
        y = np.sin(x)
        dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs[0].add_collection(lc)
        fig.colorbar(line, ax=axs[0])

        # Use a boundary norm instead
        cmap = ListedColormap(['r', 'g', 'b'])
        norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs[1].add_collection(lc)
        fig.colorbar(line, ax=axs[1])

        axs[0].set_xlim(x.min(), x.max())
        axs[0].set_ylim(-1.1, 1.1)
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_scatter_masked(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Fixing random state for reproducibility
        np.random.seed(19680801)


        N = 100
        r0 = 0.6
        x = 0.9 * np.random.rand(N)
        y = 0.9 * np.random.rand(N)
        area = (20 * np.random.rand(N))**2  # 0 to 10 point radii
        c = np.sqrt(area)
        r = np.sqrt(x ** 2 + y ** 2)
        area1 = np.ma.masked_where(r < r0, area)
        area2 = np.ma.masked_where(r >= r0, area)
        plt.scatter(x, y, s=area1, marker='^', c=c)
        plt.scatter(x, y, s=area2, marker='o', c=c)
        # Show the boundary between the regions:
        theta = np.arange(0, np.pi / 2, 0.01)
        plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_scatter_legends(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        np.random.seed(19680801)
        import matplotlib.pyplot as plt


        fig, ax = plt.subplots()
        for color in ['tab:blue', 'tab:orange', 'tab:green']:
            n = 750
            x, y = np.random.rand(2, n)
            scale = 200.0 * np.random.rand(n)
            ax.scatter(x, y, c=color, s=scale, label=color,
                    alpha=0.3, edgecolors='none')

        ax.legend()
        ax.grid(True)

        plt.show()
        N = 45
        x, y = np.random.rand(2, N)
        c = np.random.randint(1, 5, size=N)
        s = np.random.randint(10, 220, size=N)

        fig, ax = plt.subplots()

        scatter = ax.scatter(x, y, c=c, s=s)

        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Classes")
        ax.add_artist(legend1)

        # produce a legend with a cross section of sizes from the scatter
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
        legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

        plt.show()
        volume = np.random.rayleigh(27, size=40)
        amount = np.random.poisson(10, size=40)
        ranking = np.random.normal(size=40)
        price = np.random.uniform(1, 10, size=40)

        fig, ax = plt.subplots()

        # Because the price is much too small when being provided as size for ``s``,
        # we normalize it to some useful point sizes, s=0.3*(price*3)**2
        scatter = ax.scatter(volume, amount, c=ranking, s=0.3*(price*3)**2,
                            vmin=-3, vmax=3, cmap="Spectral")

        # Produce a legend for the ranking (colors). Even though there are 40 different
        # rankings, we only want to show 5 of them in the legend.
        legend1 = ax.legend(*scatter.legend_elements(num=5),
                            loc="upper left", title="Ranking")
        ax.add_artist(legend1)

        # Produce a legend for the price (sizes). Because we want to show the prices
        # in dollars, we use the *func* argument to supply the inverse of the function
        # used to calculate the sizes from above. The *fmt* ensures to show the price
        # in dollars. Note how we target at 5 elements here, but obtain only 4 in the
        # created legend due to the automatic round prices that are chosen for us.
        kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="$ {x:.2f}",
                func=lambda s: np.sqrt(s/.3)/3)
        legend2 = ax.legend(*scatter.legend_elements(**kw),
                            loc="lower right", title="Price")

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_spectrum_representation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np


        np.random.seed(0)

        dt = 0.01  # sampling interval
        Fs = 1 / dt  # sampling frequency
        t = np.arange(0, 10, dt)

        # generate noise:
        nse = np.random.randn(len(t))
        r = np.exp(-t / 0.05)
        cnse = np.convolve(nse, r) * dt
        cnse = cnse[:len(t)]

        s = 0.1 * np.sin(4 * np.pi * t) + cnse  # the signal

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

        # plot time signal:
        axs[0, 0].set_title("Signal")
        axs[0, 0].plot(t, s, color='C0')
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].set_ylabel("Amplitude")

        # plot different spectrum types:
        axs[1, 0].set_title("Magnitude Spectrum")
        axs[1, 0].magnitude_spectrum(s, Fs=Fs, color='C1')

        axs[1, 1].set_title("Log. Magnitude Spectrum")
        axs[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

        axs[2, 0].set_title("Phase Spectrum ")
        axs[2, 0].phase_spectrum(s, Fs=Fs, color='C2')

        axs[2, 1].set_title("Angle Spectrum")
        axs[2, 1].angle_spectrum(s, Fs=Fs, color='C2')

        axs[0, 1].remove()  # don't display empty ax

        fig.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_stack_and_streamplots(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        # data from United Nations World Population Prospects (Revision 2019)
        # https://population.un.org/wpp/, license: CC BY 3.0 IGO
        year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
        population_by_continent = {
            'africa': [228, 284, 365, 477, 631, 814, 1044, 1275],
            'americas': [340, 425, 519, 619, 727, 840, 943, 1006],
            'asia': [1394, 1686, 2120, 2625, 3202, 3714, 4169, 4560],
            'europe': [220, 253, 276, 295, 310, 303, 294, 293],
            'oceania': [12, 15, 19, 22, 26, 31, 36, 39],
        }

        fig, ax = plt.subplots()
        ax.stackplot(year, population_by_continent.values(),
                    labels=population_by_continent.keys())
        ax.legend(loc='upper left')
        ax.set_title('World population')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of people (millions)')

        plt.show()
        np.random.seed(19680801)


        def gaussian_mixture(x, n=5):
            def add_random_gaussian(a):
                amplitude = 1 / (.1 + np.random.random())
                dx = x[-1] - x[0]
                x0 = (2 * np.random.random() - .5) * dx
                z = 10 / (.1 + np.random.random()) / dx
                a += amplitude * np.exp(-(z * (x - x0))**2)
            a = np.zeros_like(x)
            for j in range(n):
                add_random_gaussian(a)
            return a


        x = np.linspace(0, 100, 101)
        ys = [gaussian_mixture(x) for _ in range(3)]

        fig, ax = plt.subplots()
        ax.stackplot(x, ys, baseline='wiggle')
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_step_demo(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.arange(14)
        y = np.sin(x / 2)

        plt.step(x, y + 2, label='pre (default)')
        plt.plot(x, y + 2, 'o--', color='grey', alpha=0.3)

        plt.step(x, y + 1, where='mid', label='mid')
        plt.plot(x, y + 1, 'o--', color='grey', alpha=0.3)

        plt.step(x, y, where='post', label='post')
        plt.plot(x, y, 'o--', color='grey', alpha=0.3)

        plt.grid(axis='x', color='0.95')
        plt.legend(title='Parameter where:')
        plt.title('plt.step(where=...)')
        plt.show()
        plt.plot(x, y + 2, drawstyle='steps', label='steps (=steps-pre)')
        plt.plot(x, y + 2, 'o--', color='grey', alpha=0.3)

        plt.plot(x, y + 1, drawstyle='steps-mid', label='steps-mid')
        plt.plot(x, y + 1, 'o--', color='grey', alpha=0.3)

        plt.plot(x, y, drawstyle='steps-post', label='steps-post')
        plt.plot(x, y, 'o--', color='grey', alpha=0.3)

        plt.grid(axis='x', color='0.95')
        plt.legend(title='Parameter drawstyle:')
        plt.title('plt.plot(drawstyle=...)')
        plt.show()

    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_cross_correlation_demo(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np


        # Fixing random state for reproducibility
        np.random.seed(19680801)

        x, y = np.random.randn(2, 100)
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
        ax1.xcorr(x, y, usevlines=True, maxlags=50, normed=True, lw=2)
        ax1.grid(True)

        ax2.acorr(x, usevlines=True, normed=True, maxlags=50, lw=2)
        ax2.grid(True)

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_h_and_vlines(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np


        # Fixing random state for reproducibility
        np.random.seed(19680801)

        t = np.arange(0.0, 5.0, 0.1)
        s = np.exp(-t) + np.sin(2 * np.pi * t) + 1
        nse = np.random.normal(0.0, 0.3, t.shape) * s

        fig, (vax, hax) = plt.subplots(1, 2, figsize=(12, 6))

        vax.plot(t, s + nse, '^')
        vax.vlines(t, [0], s)
        # By using ``transform=vax.get_xaxis_transform()`` the y coordinates are scaled
        # such that 0 maps to the bottom of the axes and 1 to the top.
        vax.vlines([1, 2], 0, 1, transform=vax.get_xaxis_transform(), colors='r')
        vax.set_xlabel('time (s)')
        vax.set_title('Vertical lines demo')

        hax.plot(s + nse, t, '^')
        hax.hlines(t, [0], s, lw=2)
        hax.set_xlabel('time (s)')
        hax.set_title('Horizontal lines demo')

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_timeline_dates(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.dates as mdates
        from datetime import datetime

        try:
            # Try to fetch a list of Matplotlib releases and their dates
            # from https://api.github.com/repos/matplotlib/matplotlib/releases
            import urllib.request
            import json

            url = 'https://api.github.com/repos/matplotlib/matplotlib/releases'
            url += '?per_page=100'
            data = json.loads(urllib.request.urlopen(url, timeout=.4).read().decode())

            dates = []
            names = []
            for item in data:
                if 'rc' not in item['tag_name'] and 'b' not in item['tag_name']:
                    dates.append(item['published_at'].split("T")[0])
                    names.append(item['tag_name'])
            # Convert date strings (e.g. 2014-10-18) to datetime
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

        except Exception:
            # In case the above fails, e.g. because of missing internet connection
            # use the following lists as fallback.
            names = ['v2.2.4', 'v3.0.3', 'v3.0.2', 'v3.0.1', 'v3.0.0', 'v2.2.3',
                    'v2.2.2', 'v2.2.1', 'v2.2.0', 'v2.1.2', 'v2.1.1', 'v2.1.0',
                    'v2.0.2', 'v2.0.1', 'v2.0.0', 'v1.5.3', 'v1.5.2', 'v1.5.1',
                    'v1.5.0', 'v1.4.3', 'v1.4.2', 'v1.4.1', 'v1.4.0']

            dates = ['2019-02-26', '2019-02-26', '2018-11-10', '2018-11-10',
                    '2018-09-18', '2018-08-10', '2018-03-17', '2018-03-16',
                    '2018-03-06', '2018-01-18', '2017-12-10', '2017-10-07',
                    '2017-05-10', '2017-05-02', '2017-01-17', '2016-09-09',
                    '2016-07-03', '2016-01-10', '2015-10-29', '2015-02-16',
                    '2014-10-26', '2014-10-18', '2014-08-26']

            # Convert date strings (e.g. 2014-10-18) to datetime
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    """)
    selenium.run(cmd)


def test_matplotlib_bars_lines_markers_stem_plot(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(0.1, 2 * np.pi, 41)
        y = np.exp(np.sin(x))

        plt.stem(x, y)
        plt.show()
        markerline, stemlines, baseline = plt.stem(
            x, y, linefmt='grey', markerfmt='D', bottom=1.1)
        markerline.set_markerfacecolor('none')
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_bars_lines_markers_stacked_bars(selenium, request):
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


def test_matplotlib_bars_lines_markers_CSD(selenium, request):
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
