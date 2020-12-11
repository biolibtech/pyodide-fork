from textwrap import dedent
import pytest

def test_matplotlib_subplots_axes_figures_affine_transform(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtransforms


        def get_image():
            delta = 0.25
            x = y = np.arange(-3.0, 3.0, delta)
            X, Y = np.meshgrid(x, y)
            Z1 = np.exp(-X**2 - Y**2)
            Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
            Z = (Z1 - Z2)
            return Z


        def do_plot(ax, Z, transform):
            im = ax.imshow(Z, interpolation='none',
                        origin='lower',
                        extent=[-2, 4, -3, 2], clip_on=True)

            trans_data = transform + ax.transData
            im.set_transform(trans_data)

            # display intended extent of the image
            x1, x2, y1, y2 = im.get_extent()
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
                    transform=trans_data)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-4, 4)


        # prepare image and figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        Z = get_image()

        # image rotation
        do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(30))

        # image skew
        do_plot(ax2, Z, mtransforms.Affine2D().skew_deg(30, 15))

        # scale and reflection
        do_plot(ax3, Z, mtransforms.Affine2D().scale(-1, .5))

        # everything and a translation
        do_plot(ax4, Z, mtransforms.Affine2D().
                rotate_deg(30).skew_deg(30, 15).scale(-1, .5).translate(.5, -1))

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_barcode_demo(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        # the bar
        x = np.random.rand(500) > 0.7

        barprops = dict(aspect='auto', cmap='binary', interpolation='nearest')

        fig = plt.figure()

        # a vertical barcode
        ax1 = fig.add_axes([0.1, 0.1, 0.1, 0.8])
        ax1.set_axis_off()
        ax1.imshow(x.reshape((-1, 1)), **barprops)

        # a horizontal barcode
        ax2 = fig.add_axes([0.3, 0.4, 0.6, 0.2])
        ax2.set_axis_off()
        ax2.imshow(x.reshape((1, -1)), **barprops)

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_contour_image(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import cm

        # Default delta is large because that makes it fast, and it illustrates
        # the correct registration between image and contours.
        delta = 0.5

        extent = (-3, 4, -4, 3)

        x = np.arange(-3.0, 4.001, delta)
        y = np.arange(-4.0, 3.001, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X**2 - Y**2)
        Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
        Z = (Z1 - Z2) * 2

        # Boost the upper limit to avoid truncation errors.
        levels = np.arange(-2.0, 1.601, 0.4)

        norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
        cmap = cm.PRGn

        fig, _axs = plt.subplots(nrows=2, ncols=2)
        fig.subplots_adjust(hspace=0.3)
        axs = _axs.flatten()

        cset1 = axs[0].contourf(X, Y, Z, levels, norm=norm,
                                cmap=cm.get_cmap(cmap, len(levels) - 1))
        # It is not necessary, but for the colormap, we need only the
        # number of levels minus 1.  To avoid discretization error, use
        # either this number or a large number such as the default (256).

        # If we want lines as well as filled regions, we need to call
        # contour separately; don't try to change the edgecolor or edgewidth
        # of the polygons in the collections returned by contourf.
        # Use levels output from previous call to guarantee they are the same.

        cset2 = axs[0].contour(X, Y, Z, cset1.levels, colors='k')

        # We don't really need dashed contour lines to indicate negative
        # regions, so let's turn them off.

        for c in cset2.collections:
            c.set_linestyle('solid')

        # It is easier here to make a separate call to contour than
        # to set up an array of colors and linewidths.
        # We are making a thick green line as a zero contour.
        # Specify the zero level as a tuple with only 0 in it.

        cset3 = axs[0].contour(X, Y, Z, (0,), colors='g', linewidths=2)
        axs[0].set_title('Filled contours')
        fig.colorbar(cset1, ax=axs[0])


        axs[1].imshow(Z, extent=extent, cmap=cmap, norm=norm)
        axs[1].contour(Z, levels, colors='k', origin='upper', extent=extent)
        axs[1].set_title("Image, origin 'upper'")

        axs[2].imshow(Z, origin='lower', extent=extent, cmap=cmap, norm=norm)
        axs[2].contour(Z, levels, colors='k', origin='lower', extent=extent)
        axs[2].set_title("Image, origin 'lower'")

        # We will use the interpolation "nearest" here to show the actual
        # image pixels.
        # Note that the contour lines don't extend to the edge of the box.
        # This is intentional. The Z values are defined at the center of each
        # image pixel (each color block on the following subplot), so the
        # domain that is contoured does not extend beyond these pixel centers.
        im = axs[3].imshow(Z, interpolation='nearest', extent=extent,
                        cmap=cmap, norm=norm)
        axs[3].contour(Z, levels, colors='k', origin='image', extent=extent)
        ylim = axs[3].get_ylim()
        axs[3].set_ylim(ylim[::-1])
        axs[3].set_title("Origin from rc, reversed y-axis")
        fig.colorbar(im, ax=axs[3])

        fig.tight_layout()
        plt.show()

    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_contourf(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        from numpy import ma
        from matplotlib import ticker, cm

        N = 100
        x = np.linspace(-3.0, 3.0, N)
        y = np.linspace(-2.0, 2.0, N)

        X, Y = np.meshgrid(x, y)

        # A low hump with a spike coming out.
        # Needs to have z/colour axis on a log scale so we see both hump and spike.
        # linear scale only shows the spike.
        Z1 = np.exp(-X**2 - Y**2)
        Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
        z = Z1 + 50 * Z2

        # Put in some negative values (lower left corner) to cause trouble with logs:
        z[:5, :5] = -1

        # The following is not strictly essential, but it will eliminate
        # a warning.  Comment it out to see the warning.
        z = ma.masked_where(z <= 0, z)


        # Automatic selection of levels works; setting the
        # log locator tells contourf to use a log scale:
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)

        # Alternatively, you can manually set the levels
        # and the norm:
        # lev_exp = np.arange(np.floor(np.log10(z.min())-1),
        #                    np.ceil(np.log10(z.max())+1))
        # levs = np.power(10, lev_exp)
        # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())

        cbar = fig.colorbar(cs)

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_bboximage(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.image import BboxImage
        from matplotlib.transforms import Bbox, TransformedBbox


        fig, (ax1, ax2) = plt.subplots(ncols=2)

        # ----------------------------
        # Create a BboxImage with Text
        # ----------------------------
        txt = ax1.text(0.5, 0.5, "test", size=30, ha="center", color="w")
        kwargs = dict()

        bbox_image = BboxImage(txt.get_window_extent,
                            norm=None,
                            origin=None,
                            clip_on=False,
                            **kwargs
                            )
        a = np.arange(256).reshape(1, 256)/256.
        bbox_image.set_data(a)
        ax1.add_artist(bbox_image)

        # ------------------------------------
        # Create a BboxImage for each colormap
        # ------------------------------------
        a = np.linspace(0, 1, 256).reshape(1, -1)
        a = np.vstack((a, a))

        # List of all colormaps; skip reversed colormaps.
        maps = sorted(m for m in plt.colormaps() if not m.endswith("_r"))

        ncol = 2
        nrow = len(maps)//ncol + 1

        xpad_fraction = 0.3
        dx = 1./(ncol + xpad_fraction*(ncol - 1))

        ypad_fraction = 0.3
        dy = 1./(nrow + ypad_fraction*(nrow - 1))

        for i, m in enumerate(maps):
            ix, iy = divmod(i, nrow)

            bbox0 = Bbox.from_bounds(ix*dx*(1 + xpad_fraction),
                                    1. - iy*dy*(1 + ypad_fraction) - dy,
                                    dx, dy)
            bbox = TransformedBbox(bbox0, ax2.transAxes)

            bbox_image = BboxImage(bbox,
                                cmap=plt.get_cmap(m),
                                norm=None,
                                origin=None,
                                **kwargs
                                )

            bbox_image.set_data(a)
            ax2.add_artist(bbox_image)

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_heatmap(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt

        vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                    "potato", "wheat", "barley"]
        farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
                "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

        harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                            [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                            [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                            [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                            [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                            [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                            [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


        fig, ax = plt.subplots()
        im = ax.imshow(harvest)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(farmers)))
        ax.set_yticks(np.arange(len(vegetables)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(farmers)
        ax.set_yticklabels(vegetables)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(vegetables)):
            for j in range(len(farmers)):
                text = ax.text(j, i, harvest[i, j],
                            ha="center", va="center", color="w")

        ax.set_title("Harvest of local farmers (in tons/year)")
        fig.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.arange(500) / 500 - 0.5
        y = np.arange(500) / 500 - 0.5

        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        f0 = 10
        k = 250
        a = np.sin(np.pi * 2 * (f0 * R + k * R**2 / 2))

        fig, axs = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
        for ax, interp in zip(axs, ['nearest', 'antialiased']):
            ax.imshow(a, interpolation=interp, cmap='gray')
            ax.set_title(f"interpolation='{interp}'")
        plt.show()

        fig, ax = plt.subplots(figsize=(6.8, 6.8))
        ax.imshow(a, interpolation='nearest', cmap='gray')
        ax.set_title("upsampled by factor a 1.048, interpolation='nearest'")
        plt.show()
        fig, axs = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
        for ax, interp in zip(axs, ['hanning', 'lanczos']):
            ax.imshow(a, interpolation=interp, cmap='gray')
            ax.set_title(f"interpolation='{interp}'")
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_interpolation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import matplotlib.cbook as cbook
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        # Fixing random state for reproducibility
        np.random.seed(19680801)
        delta = 0.025
        x = y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X**2 - Y**2)
        Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
        Z = (Z1 - Z2) * 2

        fig, ax = plt.subplots()
        im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
                    origin='lower', extent=[-3, 3, -3, 3],
                    vmax=abs(Z).max(), vmin=-abs(Z).max())

        plt.show()

        A = np.random.rand(5, 5)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        for ax, interp in zip(axs, ['nearest', 'bilinear', 'bicubic']):
            ax.imshow(A, interpolation=interp)
            ax.set_title(interp.capitalize())
            ax.grid(True)

        plt.show()

        x = np.arange(120).reshape((10, 12))

        interp = 'bilinear'
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))
        axs[0].set_title('blue should be up')
        axs[0].imshow(x, origin='upper', interpolation=interp)

        axs[1].set_title('blue should be down')
        axs[1].imshow(x, origin='lower', interpolation=interp)
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_delauney_tricolor(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np


        #-----------------------------------------------------------------------------
        # Analytical test function
        #-----------------------------------------------------------------------------
        def experiment_res(x, y):
            x = 2 * x
            r1 = np.sqrt((0.5 - x)**2 + (0.5 - y)**2)
            theta1 = np.arctan2(0.5 - x, 0.5 - y)
            r2 = np.sqrt((-x - 0.2)**2 + (-y - 0.2)**2)
            theta2 = np.arctan2(-x - 0.2, -y - 0.2)
            z = (4 * (np.exp((r1/10)**2) - 1) * 30 * np.cos(3 * theta1) +
                (np.exp((r2/10)**2) - 1) * 30 * np.cos(5 * theta2) +
                2 * (x**2 + y**2))
            return (np.max(z) - z) / (np.max(z) - np.min(z))

        #-----------------------------------------------------------------------------
        # Generating the initial data test points and triangulation for the demo
        #-----------------------------------------------------------------------------
        # User parameters for data test points

        # Number of test data points, tested from 3 to 5000 for subdiv=3
        n_test = 200

        # Number of recursive subdivisions of the initial mesh for smooth plots.
        # Values >3 might result in a very high number of triangles for the refine
        # mesh: new triangles numbering = (4**subdiv)*ntri
        subdiv = 3

        # Float > 0. adjusting the proportion of (invalid) initial triangles which will
        # be masked out. Enter 0 for no mask.
        init_mask_frac = 0.0

        # Minimum circle ratio - border triangles with circle ratio below this will be
        # masked if they touch a border. Suggested value 0.01; use -1 to keep all
        # triangles.
        min_circle_ratio = .01

        # Random points
        random_gen = np.random.RandomState(seed=19680801)
        x_test = random_gen.uniform(-1., 1., size=n_test)
        y_test = random_gen.uniform(-1., 1., size=n_test)
        z_test = experiment_res(x_test, y_test)

        # meshing with Delaunay triangulation
        tri = Triangulation(x_test, y_test)
        ntri = tri.triangles.shape[0]

        # Some invalid data are masked out
        mask_init = np.zeros(ntri, dtype=bool)
        masked_tri = random_gen.randint(0, ntri, int(ntri * init_mask_frac))
        mask_init[masked_tri] = True
        tri.set_mask(mask_init)


        #-----------------------------------------------------------------------------
        # Improving the triangulation before high-res plots: removing flat triangles
        #-----------------------------------------------------------------------------
        # masking badly shaped triangles at the border of the triangular mesh.
        mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
        tri.set_mask(mask)

        # refining the data
        refiner = UniformTriRefiner(tri)
        tri_refi, z_test_refi = refiner.refine_field(z_test, subdiv=subdiv)

        # analytical 'results' for comparison
        z_expected = experiment_res(tri_refi.x, tri_refi.y)

        # for the demo: loading the 'flat' triangles for plot
        flat_tri = Triangulation(x_test, y_test)
        flat_tri.set_mask(~mask)


        #-----------------------------------------------------------------------------
        # Now the plots
        #-----------------------------------------------------------------------------
        # User options for plots
        plot_tri = True          # plot of base triangulation
        plot_masked_tri = True   # plot of excessively flat excluded triangles
        plot_refi_tri = False    # plot of refined triangulation
        plot_expected = False    # plot of analytical function values for comparison


        # Graphical options for tricontouring
        levels = np.arange(0., 1., 0.025)
        cmap = cm.get_cmap(name='Blues', lut=None)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_title("Filtering a Delaunay mesh\n"
                    "(application to high-resolution tricontouring)")

        # 1) plot of the refined (computed) data contours:
        ax.tricontour(tri_refi, z_test_refi, levels=levels, cmap=cmap,
                    linewidths=[2.0, 0.5, 1.0, 0.5])
        # 2) plot of the expected (analytical) data contours (dashed):
        if plot_expected:
            ax.tricontour(tri_refi, z_expected, levels=levels, cmap=cmap,
                        linestyles='--')
        # 3) plot of the fine mesh on which interpolation was done:
        if plot_refi_tri:
            ax.triplot(tri_refi, color='0.97')
        # 4) plot of the initial 'coarse' mesh:
        if plot_tri:
            ax.triplot(tri, color='0.7')
        # 4) plot of the unvalidated triangles from naive Delaunay Triangulation:
        if plot_masked_tri:
            ax.triplot(flat_tri, color='red')

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_spectogram(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        dt = 0.0005
        t = np.arange(0.0, 20.0, dt)
        s1 = np.sin(2 * np.pi * 100 * t)
        s2 = 2 * np.sin(2 * np.pi * 400 * t)

        # create a transient "chirp"
        s2[t <= 10] = s2[12 <= t] = 0

        # add some noise into the mix
        nse = 0.01 * np.random.random(size=len(t))

        x = s1 + s2 + nse  # the signal
        NFFT = 1024  # the length of the windowing segments
        Fs = int(1.0 / dt)  # the sampling frequency

        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.plot(t, x)
        Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
        # The `specgram` method returns 4 objects. They are:
        # - Pxx: the periodogram
        # - freqs: the frequency vector
        # - bins: the centers of the time bins
        # - im: the .image.AxesImage instance representing the data in the plot
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_quadmesh(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import copy

        from matplotlib import cm, pyplot as plt
        import numpy as np

        n = 12
        x = np.linspace(-1.5, 1.5, n)
        y = np.linspace(-1.5, 1.5, n * 2)
        X, Y = np.meshgrid(x, y)
        Qx = np.cos(Y) - np.cos(X)
        Qz = np.sin(Y) + np.sin(X)
        Z = np.sqrt(X**2 + Y**2) / 5
        Z = (Z - Z.min()) / (Z.max() - Z.min())

        # The color array can include masked values.
        Zm = np.ma.masked_where(np.abs(Qz) < 0.5 * np.max(Qz), Z)

        fig, axs = plt.subplots(nrows=1, ncols=3)
        axs[0].pcolormesh(Qx, Qz, Z, shading='gouraud')
        axs[0].set_title('Without masked values')

        # You can control the color of the masked region. We copy the default colormap
        # before modifying it.
        cmap = copy.copy(cm.get_cmap(plt.rcParams['image.cmap']))
        cmap.set_bad('y', 1.0)
        axs[1].pcolormesh(Qx, Qz, Zm, shading='gouraud', cmap=cmap)
        axs[1].set_title('With masked values')

        # Or use the default, which is transparent.
        axs[2].pcolormesh(Qx, Qz, Zm, shading='gouraud')
        axs[2].set_title('With masked values')

        fig.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_pcolor(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm


        # Fixing random state for reproducibility
        np.random.seed(19680801)
        Z = np.random.rand(6, 10)

        fig, (ax0, ax1) = plt.subplots(2, 1)

        c = ax0.pcolor(Z)
        ax0.set_title('default: no edges')

        c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
        ax1.set_title('thick edges')

        fig.tight_layout()
        plt.show()
        # make these smaller to increase the resolution
        dx, dy = 0.15, 0.05

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[-3:3+dy:dy, -3:3+dx:dx]
        z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, remove the last value from the z array.
        z = z[:-1, :-1]
        z_min, z_max = -abs(z).max(), abs(z).max()

        fig, axs = plt.subplots(2, 2)

        ax = axs[0, 0]
        c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('pcolor')
        fig.colorbar(c, ax=ax)

        ax = axs[0, 1]
        c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('pcolormesh')
        fig.colorbar(c, ax=ax)

        ax = axs[1, 0]
        c = ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    interpolation='nearest', origin='lower', aspect='auto')
        ax.set_title('image (nearest, aspect="auto")')
        fig.colorbar(c, ax=ax)

        ax = axs[1, 1]
        c = ax.pcolorfast(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('pcolorfast')
        fig.colorbar(c, ax=ax)

        fig.tight_layout()
        plt.show()
        N = 100
        X, Y = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-2, 2, N))

        # A low hump with a spike coming out.
        # Needs to have z/colour axis on a log scale so we see both hump and spike.
        # linear scale only shows the spike.
        Z1 = np.exp(-X**2 - Y**2)
        Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
        Z = Z1 + 50 * Z2

        fig, (ax0, ax1) = plt.subplots(2, 1)

        c = ax0.pcolor(X, Y, Z, shading='auto',
                    norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='PuBu_r')
        fig.colorbar(c, ax=ax0)

        c = ax1.pcolor(X, Y, Z, cmap='PuBu_r', shading='auto')
        fig.colorbar(c, ax=ax1)

        plt.show()
    """)
    selenium.run(cmd)

def test_matplotlib_subplots_axes_figures_nonuniform_image(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.image import NonUniformImage
        from matplotlib import cm

        interp = 'nearest'

        # Linear x array for cell centers:
        x = np.linspace(-4, 4, 9)

        # Highly nonlinear x array:
        x2 = x**3

        y = np.linspace(-4, 4, 9)

        z = np.sqrt(x[np.newaxis, :]**2 + y[:, np.newaxis]**2)

        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        fig.suptitle('NonUniformImage class', fontsize='large')
        ax = axs[0, 0]
        im = NonUniformImage(ax, interpolation=interp, extent=(-4, 4, -4, 4),
                            cmap=cm.Purples)
        im.set_data(x, y, z)
        ax.images.append(im)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title(interp)

        ax = axs[0, 1]
        im = NonUniformImage(ax, interpolation=interp, extent=(-64, 64, -4, 4),
                            cmap=cm.Purples)
        im.set_data(x2, y, z)
        ax.images.append(im)
        ax.set_xlim(-64, 64)
        ax.set_ylim(-4, 4)
        ax.set_title(interp)

        interp = 'bilinear'

        ax = axs[1, 0]
        im = NonUniformImage(ax, interpolation=interp, extent=(-4, 4, -4, 4),
                            cmap=cm.Purples)
        im.set_data(x, y, z)
        ax.images.append(im)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title(interp)

        ax = axs[1, 1]
        im = NonUniformImage(ax, interpolation=interp, extent=(-64, 64, -4, 4),
                            cmap=cm.Purples)
        im.set_data(x2, y, z)
        ax.images.append(im)
        ax.set_xlim(-64, 64)
        ax.set_ylim(-4, 4)
        ax.set_title(interp)

        plt.show()
    """)
    selenium.run(cmd)