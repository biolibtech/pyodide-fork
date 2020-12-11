from textwrap import dedent
import pytest

def  test_matplotlib_all_the_rest_percentile_hbar(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from collections import namedtuple

        np.random.seed(42)

        Student = namedtuple('Student', ['name', 'grade', 'gender'])
        Score = namedtuple('Score', ['score', 'percentile'])

        # GLOBAL CONSTANTS
        test_names = ['Pacer Test', 'Flexed Arm\n Hang', 'Mile Run', 'Agility',
                    'Push Ups']
        test_units = dict(zip(test_names, ['laps', 'sec', 'min:sec', 'sec', '']))


        def attach_ordinal(num):
            suffixes = {str(i): v
                        for i, v in enumerate(['th', 'st', 'nd', 'rd', 'th',
                                            'th', 'th', 'th', 'th', 'th'])}
            v = str(num)
            # special case early teens
            if v in {'11', '12', '13'}:
                return v + 'th'
            return v + suffixes[v[-1]]


        def format_score(score, test):
            unit = test_units[test]
            if unit:
                return f'{score}\n{unit}'
            else:  # If no unit, don't include a newline, so that label stays centered.
                return score


        def format_ycursor(y):
            y = int(y)
            if y < 0 or y >= len(test_names):
                return ''
            else:
                return test_names[y]


        def plot_student_results(student, scores, cohort_size):
            fig, ax1 = plt.subplots(figsize=(9, 7))  # Create the figure
            fig.subplots_adjust(left=0.115, right=0.88)
            fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')

            pos = np.arange(len(test_names))

            rects = ax1.barh(pos, [scores[k].percentile for k in test_names],
                            align='center',
                            height=0.5,
                            tick_label=test_names)

            ax1.set_title(student.name)

            ax1.set_xlim([0, 100])
            ax1.xaxis.set_major_locator(MaxNLocator(11))
            ax1.xaxis.grid(True, linestyle='--', which='major',
                        color='grey', alpha=.25)

            # Plot a solid vertical gridline to highlight the median position
            ax1.axvline(50, color='grey', alpha=0.25)

            # Set the right-hand Y-axis ticks and labels
            ax2 = ax1.twinx()

            # Set the tick locations
            ax2.set_yticks(pos)
            # Set equal limits on both yaxis so that the ticks line up
            ax2.set_ylim(ax1.get_ylim())

            # Set the tick labels
            ax2.set_yticklabels([format_score(scores[k].score, k) for k in test_names])

            ax2.set_ylabel('Test Scores')

            xlabel = ('Percentile Ranking Across {grade} Grade {gender}s\n'
                    'Cohort Size: {cohort_size}')
            ax1.set_xlabel(xlabel.format(grade=attach_ordinal(student.grade),
                                        gender=student.gender.title(),
                                        cohort_size=cohort_size))

            rect_labels = []
            # Lastly, write in the ranking inside each bar to aid in interpretation
            for rect in rects:
                # Rectangle widths are already integer-valued but are floating
                # type, so it helps to remove the trailing decimal point and 0 by
                # converting width to int type
                width = int(rect.get_width())

                rank_str = attach_ordinal(width)
                # The bars aren't wide enough to print the ranking inside
                if width < 40:
                    # Shift the text to the right side of the right edge
                    xloc = 5
                    # Black against white background
                    clr = 'black'
                    align = 'left'
                else:
                    # Shift the text to the left side of the right edge
                    xloc = -5
                    # White on magenta
                    clr = 'white'
                    align = 'right'

                # Center the text vertically in the bar
                yloc = rect.get_y() + rect.get_height() / 2
                label = ax1.annotate(
                    rank_str, xy=(width, yloc), xytext=(xloc, 0),
                    textcoords="offset points",
                    horizontalalignment=align, verticalalignment='center',
                    color=clr, weight='bold', clip_on=True)
                rect_labels.append(label)

            # Make the interactive mouse over give the bar title
            ax2.fmt_ydata = format_ycursor
            # Return all of the artists created
            return {'fig': fig,
                    'ax': ax1,
                    'ax_right': ax2,
                    'bars': rects,
                    'perc_labels': rect_labels}


        student = Student('Johnny Doe', 2, 'boy')
        scores = dict(zip(
            test_names,
            (Score(v, p) for v, p in
            zip(['7', '48', '12:52', '17', '14'],
                np.round(np.random.uniform(0, 100, len(test_names)), 0)))))
        cohort_size = 62  # The number of other 2nd grade boys

        arts = plot_student_results(student, scores, cohort_size)
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_boxplot_drawer(selenium_standalone, request):
    selenium = selenium_standalone
     
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cbook as cbook

        # fake data
        np.random.seed(19680801)
        data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
        labels = list('ABCD')

        # compute the boxplot stats
        stats = cbook.boxplot_stats(data, labels=labels, bootstrap=10000)
        for n in range(len(stats)):
            stats[n]['med'] = np.median(data)
            stats[n]['mean'] *= 2

        print(list(stats[0]))

        fs = 10  # fontsize
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
        axs[0, 0].bxp(stats)
        axs[0, 0].set_title('Default', fontsize=fs)

        axs[0, 1].bxp(stats, showmeans=True)
        axs[0, 1].set_title('showmeans=True', fontsize=fs)

        axs[0, 2].bxp(stats, showmeans=True, meanline=True)
        axs[0, 2].set_title('showmeans=True,\nmeanline=True', fontsize=fs)

        axs[1, 0].bxp(stats, showbox=False, showcaps=False)
        tufte_title = 'Tufte Style\n(showbox=False,\nshowcaps=False)'
        axs[1, 0].set_title(tufte_title, fontsize=fs)

        axs[1, 1].bxp(stats, shownotches=True)
        axs[1, 1].set_title('notch=True', fontsize=fs)

        axs[1, 2].bxp(stats, showfliers=False)
        axs[1, 2].set_title('showfliers=False', fontsize=fs)

        for ax in axs.flat:
            ax.set_yscale('log')
            ax.set_yticklabels([])

        fig.subplots_adjust(hspace=0.4)
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_patchcollection(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        # Number of data points
        n = 5

        # Dummy data
        np.random.seed(19680801)
        x = np.arange(0, n, 1)
        y = np.random.rand(n) * 5.

        # Dummy errors (above and below)
        xerr = np.random.rand(2, n) + 0.1
        yerr = np.random.rand(2, n) + 0.2


        def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                            edgecolor='None', alpha=0.5):

            # Loop over data points; create box from errors at each point
            errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                        for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

            # Create patch collection with specified colour/alpha
            pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                                edgecolor=edgecolor)

            # Add collection to axes
            ax.add_collection(pc)

            # Plot errorbars
            artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                                fmt='None', ecolor='k')

            return artists


        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Call function to create error boxes
        _ = make_error_boxes(ax, x, y, xerr, yerr)

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_error_bars(selenium_standalone, request):
    selenium = selenium_standalone
     
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        # example data
        x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        y = np.exp(-x)
        xerr = 0.1
        yerr = 0.2

        # lower & upper limits of the error
        lolims = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
        uplims = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
        ls = 'dotted'

        fig, ax = plt.subplots(figsize=(7, 4))

        # standard error bars
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=ls)

        # including upper limits
        ax.errorbar(x, y + 0.5, xerr=xerr, yerr=yerr, uplims=uplims,
                    linestyle=ls)

        # including lower limits
        ax.errorbar(x, y + 1.0, xerr=xerr, yerr=yerr, lolims=lolims,
                    linestyle=ls)

        # including upper and lower limits
        ax.errorbar(x, y + 1.5, xerr=xerr, yerr=yerr,
                    lolims=lolims, uplims=uplims,
                    marker='o', markersize=8,
                    linestyle=ls)

        # Plot a series with lower and upper limits in both x & y
        # constant x-error with varying y-error
        xerr = 0.2
        yerr = np.full_like(x, 0.2)
        yerr[[3, 6]] = 0.3

        # mock up some limits by modifying previous data
        xlolims = lolims
        xuplims = uplims
        lolims = np.zeros_like(x)
        uplims = np.zeros_like(x)
        lolims[[6]] = True  # only limited at this index
        uplims[[3]] = True  # only limited at this index

        # do the plotting
        ax.errorbar(x, y + 2.1, xerr=xerr, yerr=yerr,
                    xlolims=xlolims, xuplims=xuplims,
                    uplims=uplims, lolims=lolims,
                    marker='o', markersize=8,
                    linestyle='none')

        # tidy up the figure
        ax.set_xlim((0, 5.5))
        ax.set_title('Errorbar upper and lower limits')
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_confidence_ellipse(selenium_standalone, request):
    selenium = selenium_standalone
     
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
            if x.size != y.size:
                raise ValueError("x and y must be the same size")

            cov = np.cov(x, y)
            pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
            # Using a special case to obtain the eigenvalues of this
            # two-dimensionl dataset.
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                            facecolor=facecolor, **kwargs)

            # Calculating the stdandard deviation of x from
            # the squareroot of the variance and multiplying
            # with the given number of standard deviations.
            scale_x = np.sqrt(cov[0, 0]) * n_std
            mean_x = np.mean(x)

            # calculating the stdandard deviation of y ...
            scale_y = np.sqrt(cov[1, 1]) * n_std
            mean_y = np.mean(y)

            transf = transforms.Affine2D() \
                .rotate_deg(45) \
                .scale(scale_x, scale_y) \
                .translate(mean_x, mean_y)

            ellipse.set_transform(transf + ax.transData)
            return ax.add_patch(ellipse)
        def get_correlated_dataset(n, dependency, mu, scale):
            latent = np.random.randn(n, 2)
            dependent = latent.dot(dependency)
            scaled = dependent * scale
            scaled_with_offset = scaled + mu
            # return x and y of the new, correlated dataset
            return scaled_with_offset[:, 0], scaled_with_offset[:, 1]
        np.random.seed(0)

        PARAMETERS = {
            'Positive correlation': [[0.85, 0.35],
                                    [0.15, -0.65]],
            'Negative correlation': [[0.9, -0.4],
                                    [0.1, -0.6]],
            'Weak correlation': [[1, 0],
                                [0, 1]],
        }

        mu = 2, 4
        scale = 3, 5

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        for ax, (title, dependency) in zip(axs, PARAMETERS.items()):
            x, y = get_correlated_dataset(800, dependency, mu, scale)
            ax.scatter(x, y, s=0.5)

            ax.axvline(c='grey', lw=1)
            ax.axhline(c='grey', lw=1)

            confidence_ellipse(x, y, ax, edgecolor='red')

            ax.scatter(mu[0], mu[1], c='red', s=3)
            ax.set_title(title)

        plt.show()

    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_nested_pie_charts(selenium_standalone, request):
    selenium = selenium_standalone
     
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()

        size = 0.3
        vals = np.array([[60., 32.], [37., 40.], [29., 10.]])

        cmap = plt.get_cmap("tab20c")
        outer_colors = cmap(np.arange(3)*4)
        inner_colors = cmap([1, 2, 5, 6, 9, 10])

        ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
            wedgeprops=dict(width=size, edgecolor='w'))

        ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
            wedgeprops=dict(width=size, edgecolor='w'))

        ax.set(aspect="equal", title='Pie plot with `ax.pie`')
        plt.show()

    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_polar(selenium_standalone, request):
    selenium = selenium_standalone
     
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt


        r = np.arange(0, 2, 0.01)
        theta = 2 * np.pi * r

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta, r)
        ax.set_rmax(2)
        ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)

        ax.set_title("A line plot on a polar axis", va='bottom')
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_custom_legends(selenium_standalone, request):
    selenium = selenium_standalone
     
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from matplotlib import rcParams, cycler
        import matplotlib.pyplot as plt
        import numpy as np

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        N = 10
        data = (np.geomspace(1, 10, 100) + np.random.randn(N, 100)).T
        cmap = plt.cm.coolwarm
        rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))

        fig, ax = plt.subplots()
        lines = ax.plot(data)
        ax.legend()
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(.5), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]

        fig, ax = plt.subplots()
        lines = ax.plot(data)
        ax.legend(custom_lines, ['Cold', 'Medium', 'Hot'])
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
                        Line2D([0], [0], marker='o', color='w', label='Scatter',
                                markerfacecolor='g', markersize=15),
                        Patch(facecolor='orange', edgecolor='r',
                                label='Color Patch')]

        # Create the figure
        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements, loc='center')

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_arrow(selenium_standalone, request):
    selenium = selenium_standalone
     
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np

        rates_to_bases = {'r1': 'AT', 'r2': 'TA', 'r3': 'GA', 'r4': 'AG', 'r5': 'CA',
                        'r6': 'AC', 'r7': 'GT', 'r8': 'TG', 'r9': 'CT', 'r10': 'TC',
                        'r11': 'GC', 'r12': 'CG'}
        numbered_bases_to_rates = {v: k for k, v in rates_to_bases.items()}
        lettered_bases_to_rates = {v: 'r' + v for k, v in rates_to_bases.items()}


        def make_arrow_plot(data, size=4, display='length', shape='right',
                            max_arrow_width=0.03, arrow_sep=0.02, alpha=0.5,
                            normalize_data=False, ec=None, labelcolor=None,
                            head_starts_at_zero=True,
                            rate_labels=lettered_bases_to_rates,
                            **kwargs):

            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)
            plt.gcf().set_size_inches(size, size)
            plt.xticks([])
            plt.yticks([])
            max_text_size = size * 12
            min_text_size = size
            label_text_size = size * 2.5
            text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif',
                        'fontweight': 'bold'}
            r2 = np.sqrt(2)

            deltas = {
                'AT': (1, 0),
                'TA': (-1, 0),
                'GA': (0, 1),
                'AG': (0, -1),
                'CA': (-1 / r2, 1 / r2),
                'AC': (1 / r2, -1 / r2),
                'GT': (1 / r2, 1 / r2),
                'TG': (-1 / r2, -1 / r2),
                'CT': (0, 1),
                'TC': (0, -1),
                'GC': (1, 0),
                'CG': (-1, 0)}

            colors = {
                'AT': 'r',
                'TA': 'k',
                'GA': 'g',
                'AG': 'r',
                'CA': 'b',
                'AC': 'r',
                'GT': 'g',
                'TG': 'k',
                'CT': 'b',
                'TC': 'k',
                'GC': 'g',
                'CG': 'b'}

            label_positions = {
                'AT': 'center',
                'TA': 'center',
                'GA': 'center',
                'AG': 'center',
                'CA': 'left',
                'AC': 'left',
                'GT': 'left',
                'TG': 'left',
                'CT': 'center',
                'TC': 'center',
                'GC': 'center',
                'CG': 'center'}

            def do_fontsize(k):
                return float(np.clip(max_text_size * np.sqrt(data[k]),
                                    min_text_size, max_text_size))

            plt.text(0, 1, '$A_3$', color='r', size=do_fontsize('A'), **text_params)
            plt.text(1, 1, '$T_3$', color='k', size=do_fontsize('T'), **text_params)
            plt.text(0, 0, '$G_3$', color='g', size=do_fontsize('G'), **text_params)
            plt.text(1, 0, '$C_3$', color='b', size=do_fontsize('C'), **text_params)

            arrow_h_offset = 0.25  # data coordinates, empirically determined
            max_arrow_length = 1 - 2 * arrow_h_offset
            max_head_width = 2.5 * max_arrow_width
            max_head_length = 2 * max_arrow_width
            arrow_params = {'length_includes_head': True, 'shape': shape,
                            'head_starts_at_zero': head_starts_at_zero}
            sf = 0.6  # max arrow size represents this in data coords

            d = (r2 / 2 + arrow_h_offset - 0.5) / r2  # distance for diags
            r2v = arrow_sep / r2  # offset for diags

            # tuple of x, y for start position
            positions = {
                'AT': (arrow_h_offset, 1 + arrow_sep),
                'TA': (1 - arrow_h_offset, 1 - arrow_sep),
                'GA': (-arrow_sep, arrow_h_offset),
                'AG': (arrow_sep, 1 - arrow_h_offset),
                'CA': (1 - d - r2v, d - r2v),
                'AC': (d + r2v, 1 - d + r2v),
                'GT': (d - r2v, d + r2v),
                'TG': (1 - d + r2v, 1 - d - r2v),
                'CT': (1 - arrow_sep, arrow_h_offset),
                'TC': (1 + arrow_sep, 1 - arrow_h_offset),
                'GC': (arrow_h_offset, arrow_sep),
                'CG': (1 - arrow_h_offset, -arrow_sep)}

            if normalize_data:
                # find maximum value for rates, i.e. where keys are 2 chars long
                max_val = max((v for k, v in data.items() if len(k) == 2), default=0)
                # divide rates by max val, multiply by arrow scale factor
                for k, v in data.items():
                    data[k] = v / max_val * sf

            def draw_arrow(pair, alpha=alpha, ec=ec, labelcolor=labelcolor):
                # set the length of the arrow
                if display == 'length':
                    length = (max_head_length
                            + data[pair] / sf * (max_arrow_length - max_head_length))
                else:
                    length = max_arrow_length
                # set the transparency of the arrow
                if display == 'alpha':
                    alpha = min(data[pair] / sf, alpha)

                # set the width of the arrow
                if display == 'width':
                    scale = data[pair] / sf
                    width = max_arrow_width * scale
                    head_width = max_head_width * scale
                    head_length = max_head_length * scale
                else:
                    width = max_arrow_width
                    head_width = max_head_width
                    head_length = max_head_length

                fc = colors[pair]
                ec = ec or fc

                x_scale, y_scale = deltas[pair]
                x_pos, y_pos = positions[pair]
                plt.arrow(x_pos, y_pos, x_scale * length, y_scale * length,
                        fc=fc, ec=ec, alpha=alpha, width=width,
                        head_width=head_width, head_length=head_length,
                        **arrow_params)

                # figure out coordinates for text
                # if drawing relative to base: x and y are same as for arrow
                # dx and dy are one arrow width left and up
                # need to rotate based on direction of arrow, use x_scale and y_scale
                # as sin x and cos x?
                sx, cx = y_scale, x_scale

                where = label_positions[pair]
                if where == 'left':
                    orig_position = 3 * np.array([[max_arrow_width, max_arrow_width]])
                elif where == 'absolute':
                    orig_position = np.array([[max_arrow_length / 2.0,
                                            3 * max_arrow_width]])
                elif where == 'right':
                    orig_position = np.array([[length - 3 * max_arrow_width,
                                            3 * max_arrow_width]])
                elif where == 'center':
                    orig_position = np.array([[length / 2.0, 3 * max_arrow_width]])
                else:
                    raise ValueError("Got unknown position parameter %s" % where)

                M = np.array([[cx, sx], [-sx, cx]])
                coords = np.dot(orig_position, M) + [[x_pos, y_pos]]
                x, y = np.ravel(coords)
                orig_label = rate_labels[pair]
                label = r'$%s_{_{\mathrm{%s}}}$' % (orig_label[0], orig_label[1:])

                plt.text(x, y, label, size=label_text_size, ha='center', va='center',
                        color=labelcolor or fc)

            for p in sorted(positions):
                draw_arrow(p)


        # test data
        all_on_max = dict([(i, 1) for i in 'TCAG'] +
                        [(i + j, 0.6) for i in 'TCAG' for j in 'TCAG'])

        realistic_data = {
            'A': 0.4,
            'T': 0.3,
            'G': 0.5,
            'C': 0.2,
            'AT': 0.4,
            'AC': 0.3,
            'AG': 0.2,
            'TA': 0.2,
            'TC': 0.3,
            'TG': 0.4,
            'CT': 0.2,
            'CG': 0.3,
            'CA': 0.2,
            'GA': 0.1,
            'GT': 0.4,
            'GC': 0.1}

        extreme_data = {
            'A': 0.75,
            'T': 0.10,
            'G': 0.10,
            'C': 0.05,
            'AT': 0.6,
            'AC': 0.3,
            'AG': 0.1,
            'TA': 0.02,
            'TC': 0.3,
            'TG': 0.01,
            'CT': 0.2,
            'CG': 0.5,
            'CA': 0.2,
            'GA': 0.1,
            'GT': 0.4,
            'GC': 0.2}

        sample_data = {
            'A': 0.2137,
            'T': 0.3541,
            'G': 0.1946,
            'C': 0.2376,
            'AT': 0.0228,
            'AC': 0.0684,
            'AG': 0.2056,
            'TA': 0.0315,
            'TC': 0.0629,
            'TG': 0.0315,
            'CT': 0.1355,
            'CG': 0.0401,
            'CA': 0.0703,
            'GA': 0.1824,
            'GT': 0.0387,
            'GC': 0.1106}


        if __name__ == '__main__':
            from sys import argv
            d = None
            if len(argv) > 1:
                if argv[1] == 'full':
                    d = all_on_max
                    scaled = False
                elif argv[1] == 'extreme':
                    d = extreme_data
                    scaled = False
                elif argv[1] == 'realistic':
                    d = realistic_data
                    scaled = False
                elif argv[1] == 'sample':
                    d = sample_data
                    scaled = True
            if d is None:
                d = all_on_max
                scaled = False
            if len(argv) > 2:
                display = argv[2]
            else:
                display = 'length'

            size = 4
            plt.figure(figsize=(size, size))

            make_arrow_plot(d, display=display, linewidth=0.001, edgecolor=None,
                            normalize_data=scaled, head_starts_at_zero=True, size=size)

            plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_fancy_arrow(selenium_standalone, request):
    selenium = selenium_standalone
     
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        styles = mpatches.ArrowStyle.get_styles()

        ncol = 2
        nrow = (len(styles) + 1) // ncol
        figheight = (nrow + 0.5)
        fig = plt.figure(figsize=(4 * ncol / 1.5, figheight / 1.5))
        fontsize = 0.2 * 70


        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)

        ax.set_xlim(0, 4 * ncol)
        ax.set_ylim(0, figheight)


        def to_texstring(s):
            s = s.replace("<", r"$<$")
            s = s.replace(">", r"$>$")
            s = s.replace("|", r"$|$")
            return s


        for i, (stylename, styleclass) in enumerate(sorted(styles.items())):
            x = 3.2 + (i // nrow) * 4
            y = (figheight - 0.7 - i % nrow)  # /figheight
            p = mpatches.Circle((x, y), 0.2)
            ax.add_patch(p)

            ax.annotate(to_texstring(stylename), (x, y),
                        (x - 1.2, y),
                        ha="right", va="center",
                        size=fontsize,
                        arrowprops=dict(arrowstyle=stylename,
                                        patchB=p,
                                        shrinkA=5,
                                        shrinkB=5,
                                        fc="k", ec="k",
                                        connectionstyle="arc3,rad=-0.05",
                                        ),
                        bbox=dict(boxstyle="square", fc="w"))

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_text_rotation_relative_to_line(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()

        # Plot diagonal line (45 degrees)
        h = ax.plot(range(0, 10), range(0, 10))

        # set limits so that it no longer looks on screen to be 45 degrees
        ax.set_xlim([-10, 20])

        # Locations to plot text
        l1 = np.array((1, 1))
        l2 = np.array((5, 5))

        # Rotate angle
        angle = 45
        trans_angle = ax.transData.transform_angles([45], l2.reshape((1, 2)))[0]

        # Plot text
        th1 = ax.text(*l1, 'text not rotated correctly', fontsize=16,
                    rotation=angle, rotation_mode='anchor')
        th2 = ax.text(*l2, 'text rotated correctly', fontsize=16,
                    rotation=trans_angle, rotation_mode='anchor')

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_align_y_labels(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt


        def make_plot(axs):
            box = dict(facecolor='yellow', pad=5, alpha=0.2)

            # Fixing random state for reproducibility
            np.random.seed(19680801)
            ax1 = axs[0, 0]
            ax1.plot(2000*np.random.rand(10))
            ax1.set_title('ylabels not aligned')
            ax1.set_ylabel('misaligned 1', bbox=box)
            ax1.set_ylim(0, 2000)

            ax3 = axs[1, 0]
            ax3.set_ylabel('misaligned 2', bbox=box)
            ax3.plot(np.random.rand(10))

            ax2 = axs[0, 1]
            ax2.set_title('ylabels aligned')
            ax2.plot(2000*np.random.rand(10))
            ax2.set_ylabel('aligned 1', bbox=box)
            ax2.set_ylim(0, 2000)

            ax4 = axs[1, 1]
            ax4.plot(np.random.rand(10))
            ax4.set_ylabel('aligned 2', bbox=box)


        # Plot 1:
        fig, axs = plt.subplots(2, 2)
        fig.subplots_adjust(left=0.2, wspace=0.6)
        make_plot(axs)

        # just align the last column of axes:
        fig.align_ylabels(axs[:, 1])
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_infinite_lines(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(-10, 10, 100)
        sig = 1 / (1 + np.exp(-t))

        plt.axhline(y=0, color="black", linestyle="--")
        plt.axhline(y=0.5, color="black", linestyle=":")
        plt.axhline(y=1.0, color="black", linestyle="--")
        plt.axvline(color="grey")
        plt.axline((0, 0.5), slope=0.25, color="black", linestyle=(0, (5, 5)))
        plt.plot(t, sig, linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
        plt.xlim(-10, 10)
        plt.xlabel("t")
        plt.legend(fontsize=14)
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_pyplt_two_subplots(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt


        def f(t):
            return np.exp(-t) * np.cos(2*np.pi*t)


        t1 = np.arange(0.0, 5.0, 0.1)
        t2 = np.arange(0.0, 5.0, 0.02)

        plt.figure()
        plt.subplot(211)
        plt.plot(t1, f(t1), color='tab:blue', marker='o')
        plt.plot(t2, f(t2), color='black')

        plt.subplot(212)
        plt.plot(t2, np.cos(2*np.pi*t2), color='tab:orange', linestyle='--')
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_colorbar(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        # setup some generic data
        N = 37
        x, y = np.mgrid[:N, :N]
        Z = (np.cos(x*0.2) + np.sin(y*0.3))

        # mask out the negative and positive values, respectively
        Zpos = np.ma.masked_less(Z, 0)
        Zneg = np.ma.masked_greater(Z, 0)

        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

        # plot just the positive data and save the
        # color "mappable" object returned by ax1.imshow
        pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')

        # add the colorbar using the figure's method,
        # telling which mappable we're talking about and
        # which axes object it should be near
        fig.colorbar(pos, ax=ax1)

        # repeat everything above for the negative data
        neg = ax2.imshow(Zneg, cmap='Reds_r', interpolation='none')
        fig.colorbar(neg, ax=ax2)

        # Plot both positive and negative values between +/- 1.2
        pos_neg_clipped = ax3.imshow(Z, cmap='RdBu', vmin=-1.2, vmax=1.2,
                                    interpolation='none')
        # Add minorticks on the colorbar to make it easy to read the
        # values off the colorbar.
        cbar = fig.colorbar(pos_neg_clipped, ax=ax3, extend='both')
        cbar.minorticks_on()
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_hatch(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Polygon

        fig, (ax1, ax2, ax3) = plt.subplots(3)

        ax1.bar(range(1, 5), range(1, 5), color='red', edgecolor='black', hatch="/")
        ax1.bar(range(1, 5), [6] * 4, bottom=range(1, 5),
                color='blue', edgecolor='black', hatch='//')
        ax1.set_xticks([1.5, 2.5, 3.5, 4.5])

        bars = ax2.bar(range(1, 5), range(1, 5), color='yellow', ecolor='black') + \
            ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5),
                    color='green', ecolor='black')
        ax2.set_xticks([1.5, 2.5, 3.5, 4.5])

        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)

        ax3.fill([1, 3, 3, 1], [1, 1, 2, 2], fill=False, hatch='\\')
        ax3.add_patch(Ellipse((4, 1.5), 4, 0.5, fill=False, hatch='*'))
        ax3.add_patch(Polygon([[0, 0], [4, 1.1], [6, 2.5], [2, 1.4]], closed=True,
                            fill=False, hatch='/'))
        ax3.set_xlim((0, 6))
        ax3.set_ylim((0, 2.5))

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_shapes_for_artists(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.path as mpath
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection


        def label(xy, text):
            y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
            plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)


        fig, ax = plt.subplots()
        # create 3x3 grid to plot the artists
        grid = np.mgrid[0.2:0.8:3j, 0.2:0.8:3j].reshape(2, -1).T

        patches = []

        # add a circle
        circle = mpatches.Circle(grid[0], 0.1, ec="none")
        patches.append(circle)
        label(grid[0], "Circle")

        # add a rectangle
        rect = mpatches.Rectangle(grid[1] - [0.025, 0.05], 0.05, 0.1, ec="none")
        patches.append(rect)
        label(grid[1], "Rectangle")

        # add a wedge
        wedge = mpatches.Wedge(grid[2], 0.1, 30, 270, ec="none")
        patches.append(wedge)
        label(grid[2], "Wedge")

        # add a Polygon
        polygon = mpatches.RegularPolygon(grid[3], 5, 0.1)
        patches.append(polygon)
        label(grid[3], "Polygon")

        # add an ellipse
        ellipse = mpatches.Ellipse(grid[4], 0.2, 0.1)
        patches.append(ellipse)
        label(grid[4], "Ellipse")

        # add an arrow
        arrow = mpatches.Arrow(grid[5, 0] - 0.05, grid[5, 1] - 0.05, 0.1, 0.1,
                            width=0.1)
        patches.append(arrow)
        label(grid[5], "Arrow")

        # add a path patch
        Path = mpath.Path
        path_data = [
            (Path.MOVETO, [0.018, -0.11]),
            (Path.CURVE4, [-0.031, -0.051]),
            (Path.CURVE4, [-0.115, 0.073]),
            (Path.CURVE4, [-0.03, 0.073]),
            (Path.LINETO, [-0.011, 0.039]),
            (Path.CURVE4, [0.043, 0.121]),
            (Path.CURVE4, [0.075, -0.005]),
            (Path.CURVE4, [0.035, -0.027]),
            (Path.CLOSEPOLY, [0.018, -0.11])]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts + grid[6], codes)
        patch = mpatches.PathPatch(path)
        patches.append(patch)
        label(grid[6], "PathPatch")

        # add a fancy box
        fancybox = mpatches.FancyBboxPatch(
            grid[7] - [0.025, 0.05], 0.05, 0.1,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02))
        patches.append(fancybox)
        label(grid[7], "FancyBboxPatch")

        # add a line
        x, y = ([-0.06, 0.0, 0.1], [0.05, -0.05, 0.05])
        line = mlines.Line2D(x + grid[8, 0], y + grid[8, 1], lw=5., alpha=0.3)
        label(grid[8], "Line2D")

        colors = np.linspace(0, 1, len(patches))
        collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
        collection.set_array(colors)
        ax.add_collection(collection)
        ax.add_line(line)

        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_donut_shapes(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.path as mpath
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt


        def wise(v):
            if v == 1:
                return "CCW"
            else:
                return "CW"


        def make_circle(r):
            t = np.arange(0, np.pi * 2.0, 0.01)
            t = t.reshape((len(t), 1))
            x = r * np.cos(t)
            y = r * np.sin(t)
            return np.hstack((x, y))

        Path = mpath.Path

        fig, ax = plt.subplots()

        inside_vertices = make_circle(0.5)
        outside_vertices = make_circle(1.0)
        codes = np.ones(
            len(inside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
        codes[0] = mpath.Path.MOVETO

        for i, (inside, outside) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            # Concatenate the inside and outside subpaths together, changing their
            # order as needed
            vertices = np.concatenate((outside_vertices[::outside],
                                    inside_vertices[::inside]))
            # Shift the path
            vertices[:, 0] += i * 2.5
            # The codes will be all "LINETO" commands, except for "MOVETO"s at the
            # beginning of each subpath
            all_codes = np.concatenate((codes, codes))
            # Create the Path object
            path = mpath.Path(vertices, all_codes)
            # Add plot it
            patch = mpatches.PathPatch(path, facecolor='#885500', edgecolor='black')
            ax.add_patch(patch)

            ax.annotate("Outside %s,\nInside %s" % (wise(outside), wise(inside)),
                        (i * 2.5, -1.5), va="top", ha="center")

        ax.set_xlim(-2, 10)
        ax.set_ylim(-3, 2)
        ax.set_title('Mmm, donuts!')
        ax.set_aspect(1.0)
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_ggplot_stylesheet(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('ggplot')

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        fig, axs = plt.subplots(ncols=2, nrows=2)
        ax1, ax2, ax3, ax4 = axs.ravel()

        # scatter plot (Note: `plt.scatter` doesn't use default colors)
        x, y = np.random.normal(size=(2, 200))
        ax1.plot(x, y, 'o')

        # sinusoidal lines with colors from default color cycle
        L = 2*np.pi
        x = np.linspace(0, L)
        ncolors = len(plt.rcParams['axes.prop_cycle'])
        shift = np.linspace(0, L, ncolors, endpoint=False)
        for s in shift:
            ax2.plot(x, np.sin(x + s), '-')
        ax2.margins(0)

        # bar graphs
        x = np.arange(5)
        y1, y2 = np.random.randint(1, 25, size=(2, 5))
        width = 0.25
        ax3.bar(x, y1, width)
        ax3.bar(x + width, y2, width,
                color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(['a', 'b', 'c', 'd', 'e'])

        # circles with colors from default color cycle
        for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
            xy = np.random.normal(size=2)
            ax4.add_patch(plt.Circle(xy, radius=0.3, color=color['color']))
        ax4.axis('equal')
        ax4.margins(0)

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_538_stylesheet(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np


        plt.style.use('fivethirtyeight')

        x = np.linspace(0, 10)

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        fig, ax = plt.subplots()

        ax.plot(x, np.sin(x) + x + np.random.randn(50))
        ax.plot(x, np.sin(x) + 0.5 * x + np.random.randn(50))
        ax.plot(x, np.sin(x) + 2 * x + np.random.randn(50))
        ax.plot(x, np.sin(x) - 0.5 * x + np.random.randn(50))
        ax.plot(x, np.sin(x) - 2 * x + np.random.randn(50))
        ax.plot(x, np.sin(x) + np.random.randn(50))
        ax.set_title("'fivethirtyeight' style sheet")

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_axes_divider(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from matplotlib import cbook
        import matplotlib.pyplot as plt


        def get_demo_image():
            z = cbook.get_sample_data("axes_grid/bivariate_normal.npy", np_load=True)
            # z is a numpy array of 15x15
            return z, (-3, 4, -4, 3)


        def demo_simple_image(ax):
            Z, extent = get_demo_image()

            im = ax.imshow(Z, extent=extent)
            cb = plt.colorbar(im)
            plt.setp(cb.ax.get_yticklabels(), visible=False)


        def demo_locatable_axes_hard(fig):

            from mpl_toolkits.axes_grid1 import SubplotDivider, Size
            from mpl_toolkits.axes_grid1.mpl_axes import Axes

            divider = SubplotDivider(fig, 2, 2, 2, aspect=True)

            # axes for image
            ax = Axes(fig, divider.get_position())

            # axes for colorbar
            ax_cb = Axes(fig, divider.get_position())

            h = [Size.AxesX(ax),  # main axes
                Size.Fixed(0.05),  # padding, 0.1 inch
                Size.Fixed(0.2),  # colorbar, 0.3 inch
                ]

            v = [Size.AxesY(ax)]

            divider.set_horizontal(h)
            divider.set_vertical(v)

            ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
            ax_cb.set_axes_locator(divider.new_locator(nx=2, ny=0))

            fig.add_axes(ax)
            fig.add_axes(ax_cb)

            ax_cb.axis["left"].toggle(all=False)
            ax_cb.axis["right"].toggle(ticks=True)

            Z, extent = get_demo_image()

            im = ax.imshow(Z, extent=extent)
            plt.colorbar(im, cax=ax_cb)
            plt.setp(ax_cb.get_yticklabels(), visible=False)


        def demo_locatable_axes_easy(ax):
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)

            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            fig = ax.get_figure()
            fig.add_axes(ax_cb)

            Z, extent = get_demo_image()
            im = ax.imshow(Z, extent=extent)

            plt.colorbar(im, cax=ax_cb)
            ax_cb.yaxis.tick_right()
            ax_cb.yaxis.set_tick_params(labelright=False)


        def demo_images_side_by_side(ax):
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)

            Z, extent = get_demo_image()
            ax2 = divider.new_horizontal(size="100%", pad=0.05)
            fig1 = ax.get_figure()
            fig1.add_axes(ax2)

            ax.imshow(Z, extent=extent)
            ax2.imshow(Z, extent=extent)
            ax2.yaxis.set_tick_params(labelleft=False)


        def demo():

            fig = plt.figure(figsize=(6, 6))

            # PLOT 1
            # simple image & colorbar
            ax = fig.add_subplot(2, 2, 1)
            demo_simple_image(ax)

            # PLOT 2
            # image and colorbar whose location is adjusted in the drawing time.
            # a hard way

            demo_locatable_axes_hard(fig)

            # PLOT 3
            # image and colorbar whose location is adjusted in the drawing time.
            # a easy way

            ax = fig.add_subplot(2, 2, 3)
            demo_locatable_axes_easy(ax)

            # PLOT 4
            # two images side by side with fixed padding.

            ax = fig.add_subplot(2, 2, 4)
            demo_images_side_by_side(ax)

            plt.show()
        demo()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_hboxdivider(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider
        import mpl_toolkits.axes_grid1.axes_size as Size


        def make_heights_equal(fig, rect, ax1, ax2, pad):
            # pad in inches
            divider = HBoxDivider(
                fig, rect,
                horizontal=[Size.AxesX(ax1), Size.Fixed(pad), Size.AxesX(ax2)],
                vertical=[Size.AxesY(ax1), Size.Scaled(1), Size.AxesY(ax2)])
            ax1.set_axes_locator(divider.new_locator(0))
            ax2.set_axes_locator(divider.new_locator(2))


        if __name__ == "__main__":

            arr1 = np.arange(20).reshape((4, 5))
            arr2 = np.arange(20).reshape((5, 4))

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(arr1)
            ax2.imshow(arr2)

            make_heights_equal(fig, 111, ax1, ax2, pad=0.5)

            fig.text(.5, .5,
                    "Both axes' location are adjusted\n"
                    "so that they have equal heights\n"
                    "while maintaining their aspect ratios",
                    va="center", ha="center",
                    bbox=dict(boxstyle="round, pad=1", facecolor="w"))

            plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_scatter_histogram(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        # the random data
        x = np.random.randn(1000)
        y = np.random.randn(1000)


        fig, ax = plt.subplots(figsize=(5.5, 5.5))

        # the scatter plot:
        ax.scatter(x, y)

        # Set aspect of the main axes.
        ax.set_aspect(1.)

        # create new axes on the right and on the top of the current axes
        divider = make_axes_locatable(ax)
        # below height and pad are in inches
        ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
        ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

        # make some labels invisible
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1)*binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

        # the xaxis of ax_histx and yaxis of ax_histy are shared with ax,
        # thus there is no need to manually adjust the xlim and ylim of these
        # axis.

        ax_histx.set_yticks([0, 50, 100])
        ax_histy.set_xticks([0, 50, 100])

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_image_grid(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
        import numpy as np

        im1 = np.arange(100).reshape((10, 10))
        im2 = im1.T
        im3 = np.flipud(im1)
        im4 = np.fliplr(im2)

        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        for ax, im in zip(grid, [im1, im2, im3, im4]):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_direction_step1(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import mpl_toolkits.axisartist as axisartist


        def setup_axes(fig, rect):
            ax = axisartist.Subplot(fig, rect)
            fig.add_axes(ax)

            ax.set_ylim(-0.1, 1.5)
            ax.set_yticks([0, 1])

            ax.axis[:].set_visible(False)

            ax.axis["x"] = ax.new_floating_axis(1, 0.5)
            ax.axis["x"].set_axisline_style("->", size=1.5)

            return ax


        fig = plt.figure(figsize=(3, 2.5))
        fig.subplots_adjust(top=0.8)
        ax1 = setup_axes(fig, 111)

        ax1.axis["x"].set_axis_direction("left")

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_parasite_axes(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
        import matplotlib.pyplot as plt


        fig = plt.figure()

        host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
        par1 = ParasiteAxes(host, sharex=host)
        par2 = ParasiteAxes(host, sharex=host)
        host.parasites.append(par1)
        host.parasites.append(par2)

        host.axis["right"].set_visible(False)

        par1.axis["right"].set_visible(True)
        par1.axis["right"].major_ticklabels.set_visible(True)
        par1.axis["right"].label.set_visible(True)

        par2.axis["right2"] = par2.new_fixed_axis(loc="right", offset=(60, 0))

        fig.add_axes(host)

        p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
        p2, = par1.plot([0, 1, 2], [0, 3, 2], label="Temperature")
        p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")

        host.set_xlim(0, 2)
        host.set_ylim(0, 2)
        par1.set_ylim(0, 4)
        par2.set_ylim(1, 65)

        host.set_xlabel("Distance")
        host.set_ylabel("Density")
        par1.set_ylabel("Temperature")
        par2.set_ylabel("Velocity")

        host.legend()

        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
        par2.axis["right2"].label.set_color(p3.get_color())

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_decay(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import itertools

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation


        def data_gen():
            for cnt in itertools.count():
                t = cnt / 10
                yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)


        def init():
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlim(0, 10)
            del xdata[:]
            del ydata[:]
            line.set_data(xdata, ydata)
            return line,

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.grid()
        xdata, ydata = [], []


        def run(data):
            # update the data
            t, y = data
            xdata.append(t)
            ydata.append(y)
            xmin, xmax = ax.get_xlim()

            if t >= xmax:
                ax.set_xlim(xmin, 2*xmax)
                ax.figure.canvas.draw()
            line.set_data(xdata, ydata)

            return line,

        ani = animation.FuncAnimation(fig, run, data_gen, interval=2, init_func=init)
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_3d_bar_chart(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt


        # setup the figure and axes
        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # fake data
        _x = np.arange(4)
        _y = np.arange(5)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        top = x + y
        bottom = np.zeros_like(top)
        width = depth = 1

        ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
        ax1.set_title('Shaded')

        ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
        ax2.set_title('Not Shaded')

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_lorenz_attractor(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt


        def lorenz(x, y, z, s=10, r=28, b=2.667):
            x_dot = s*(y - x)
            y_dot = r*x - y - x*z
            z_dot = x*y - b*z
            return x_dot, y_dot, z_dot


        dt = 0.01
        num_steps = 10000

        # Need one more for the initial values
        xs = np.empty(num_steps + 1)
        ys = np.empty(num_steps + 1)
        zs = np.empty(num_steps + 1)

        # Set initial values
        xs[0], ys[0], zs[0] = (0., 1., 1.05)

        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)


        # Plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(xs, ys, zs, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_auto_tick_labels(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        np.random.seed(19680801)

        fig, ax = plt.subplots()
        dots = np.arange(10) / 100. + .03
        x, y = np.meshgrid(dots, dots)
        data = [x.ravel(), y.ravel()]
        ax.scatter(*data, c=data[1])
        print(plt.rcParams['axes.autolimit_mode'])

        # Now change this value and see the results
        with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):
            fig, ax = plt.subplots()
            ax.scatter(*data, c=data[1])
        with plt.rc_context({'axes.autolimit_mode': 'round_numbers',
                     'axes.xmargin': .8,
                     'axes.ymargin': .8}):
        fig, ax = plt.subplots()
        ax.scatter(*data, c=data[1])

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_colorbar_tick(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import cm
        from numpy.random import randn


        # Fixing random state for reproducibility
        np.random.seed(19680801)
        fig, ax = plt.subplots()

        data = np.clip(randn(250, 250), -1, 1)

        cax = ax.imshow(data, cmap=cm.coolwarm)
        ax.set_title('Gaussian noise with vertical colorbar')

        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['< -1', '0', '> 1']) 
        fig, ax = plt.subplots()

        data = np.clip(randn(250, 250), -1, 1)

        cax = ax.imshow(data, cmap=cm.afmhot)
        ax.set_title('Gaussian noise with horizontal colorbar')

        cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
        cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_date_precision_and_epoch(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import datetime
        import numpy as np

        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates


        def _reset_epoch_for_tutorial():
            mdates._reset_epoch_test_example()

        old_epoch = '0000-12-31T00:00:00'
        new_epoch = '1970-01-01T00:00:00'

        _reset_epoch_for_tutorial()  # Don't do this.  Just for this tutorial.
        mdates.set_epoch(old_epoch)  # old epoch (pre MPL 3.3)

        date1 = datetime.datetime(2000, 1, 1, 0, 10, 0, 12,
                                tzinfo=datetime.timezone.utc)
        mdate1 = mdates.date2num(date1)
        print('Before Roundtrip: ', date1, 'Matplotlib date:', mdate1)
        date2 = mdates.num2date(mdate1)
        print('After Roundtrip:  ', date2)
        old_epoch = '0000-12-31T00:00:00'
        new_epoch = '1970-01-01T00:00:00'

        _reset_epoch_for_tutorial()  # Don't do this.  Just for this tutorial.
        mdates.set_epoch(old_epoch)  # old epoch (pre MPL 3.3)

        date1 = datetime.datetime(2000, 1, 1, 0, 10, 0, 12,
                                tzinfo=datetime.timezone.utc)
        mdate1 = mdates.date2num(date1)
        print('Before Roundtrip: ', date1, 'Matplotlib date:', mdate1)
        date2 = mdates.num2date(mdate1)
        print('After Roundtrip:  ', date2)

    """)

    selenium.run(cmd)

def  test_matplotlib_all_the_rest_date_index_formatter(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import dateutil.parser
        from matplotlib import cbook, dates
        import matplotlib.pyplot as plt
        from matplotlib.ticker import Formatter
        import numpy as np


        datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
        print('loading %s' % datafile)
        msft_data = np.genfromtxt(
            datafile, delimiter=',', names=True,
            converters={0: lambda s: dates.date2num(dateutil.parser.parse(s))})


        class MyFormatter(Formatter):
            def __init__(self, dates, fmt='%Y-%m-%d'):
                self.dates = dates
                self.fmt = fmt

            def __call__(self, x, pos=0):
                ind = int(round(x))
                if ind >= len(self.dates) or ind < 0:
                    return ''
                return dates.num2date(self.dates[ind]).strftime(self.fmt)


        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(MyFormatter(msft_data['Date']))
        ax.plot(msft_data['Close'], 'o-')
        fig.autofmt_xdate()
        plt.show()
    """)
    selenium.run(cmd)

def  test_matplotlib_all_the_rest_tick_locators(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker


        def setup(ax, title):
            # only show the bottom spine
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')

            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(which='major', width=1.00, length=5)
            ax.tick_params(which='minor', width=0.75, length=2.5)
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1)
            ax.text(0.0, 0.2, title, transform=ax.transAxes,
                    fontsize=14, fontname='Monospace', color='tab:blue')


        fig, axs = plt.subplots(8, 1, figsize=(8, 6))

        # Null Locator
        setup(axs[0], title="NullLocator()")
        axs[0].xaxis.set_major_locator(ticker.NullLocator())
        axs[0].xaxis.set_minor_locator(ticker.NullLocator())

        # Multiple Locator
        setup(axs[1], title="MultipleLocator(0.5)")
        axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        # Fixed Locator
        setup(axs[2], title="FixedLocator([0, 1, 5])")
        axs[2].xaxis.set_major_locator(ticker.FixedLocator([0, 1, 5]))
        axs[2].xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(0.2, 0.8, 4)))

        # Linear Locator
        setup(axs[3], title="LinearLocator(numticks=3)")
        axs[3].xaxis.set_major_locator(ticker.LinearLocator(3))
        axs[3].xaxis.set_minor_locator(ticker.LinearLocator(31))

        # Index Locator
        setup(axs[4], title="IndexLocator(base=0.5, offset=0.25)")
        axs[4].plot(range(0, 5), [0]*5, color='white')
        axs[4].xaxis.set_major_locator(ticker.IndexLocator(base=0.5, offset=0.25))

        # Auto Locator
        setup(axs[5], title="AutoLocator()")
        axs[5].xaxis.set_major_locator(ticker.AutoLocator())
        axs[5].xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # MaxN Locator
        setup(axs[6], title="MaxNLocator(n=4)")
        axs[6].xaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[6].xaxis.set_minor_locator(ticker.MaxNLocator(40))

        # Log Locator
        setup(axs[7], title="LogLocator(base=10, numticks=15)")
        axs[7].set_xlim(10**3, 10**10)
        axs[7].set_xscale('log')
        axs[7].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)
