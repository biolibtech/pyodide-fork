from textwrap import dedent
import pytest

def test_sklearn_decomposition_beta_divergence(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition.nmf import _beta_divergence

        print(__doc__)

        x = np.linspace(0.001, 4, 1000)
        y = np.zeros(x.shape)

        colors = 'mbgyr'
        for j, beta in enumerate((0., 0.5, 1., 1.5, 2.)):
            for i, xi in enumerate(x):
                y[i] = _beta_divergence(1, xi, 1, beta)
            name = "beta = %1.1f" % beta
            plt.plot(x, y, label=name, color=colors[j])

        plt.xlabel("x")
        plt.title("beta-divergence(1, x)")
        plt.legend(loc=0)
        plt.axis([0, 4, 0, 3])
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_decomposition_PCA(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D


        from sklearn import decomposition
        from sklearn import datasets

        np.random.seed(5)

        centers = [[1, 1], [-1, -1], [1, -1]]
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)

        for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
            ax.text3D(X[y == label, 0].mean(),
                    X[y == label, 1].mean() + 1.5,
                    X[y == label, 2].mean(), name,
                    horizontalalignment='center',
                    bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
                edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_decomposition_FastICA(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import signal

        from sklearn.decomposition import FastICA, PCA

        # #############################################################################
        # Generate sample data
        np.random.seed(0)
        n_samples = 2000
        time = np.linspace(0, 8, n_samples)

        s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
        s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

        S = np.c_[s1, s2, s3]
        S += 0.2 * np.random.normal(size=S.shape)  # Add noise

        S /= S.std(axis=0)  # Standardize data
        # Mix data
        A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
        X = np.dot(S, A.T)  # Generate observations

        # Compute ICA
        ica = FastICA(n_components=3)
        S_ = ica.fit_transform(X)  # Reconstruct signals
        A_ = ica.mixing_  # Get estimated mixing matrix

        # We can `prove` that the ICA model applies by reverting the unmixing.
        assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

        # For comparison, compute PCA
        pca = PCA(n_components=3)
        H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

        # #############################################################################
        # Plot results

        plt.figure()

        models = [X, S, S_, H]
        names = ['Observations (mixed signal)',
                'True Sources',
                'ICA recovered signals',
                'PCA recovered signals']
        colors = ['red', 'steelblue', 'orange']

        for ii, (model, name) in enumerate(zip(models, names), 1):
            plt.subplot(4, 1, ii)
            plt.title(name)
            for sig, color in zip(model.T, colors):
                plt.plot(sig, color=color)

        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_decomposition_kernel_PCA(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.decomposition import PCA, KernelPCA
        from sklearn.datasets import make_circles

        np.random.seed(0)

        X, y = make_circles(n_samples=400, factor=.3, noise=.05)

        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        X_kpca = kpca.fit_transform(X)
        X_back = kpca.inverse_transform(X_kpca)
        pca = PCA()
        X_pca = pca.fit_transform(X)

        # Plot results

        plt.figure()
        plt.subplot(2, 2, 1, aspect='equal')
        plt.title("Original space")
        reds = y == 0
        blues = y == 1

        plt.scatter(X[reds, 0], X[reds, 1], c="red",
                    s=20, edgecolor='k')
        plt.scatter(X[blues, 0], X[blues, 1], c="blue",
                    s=20, edgecolor='k')
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

        X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
        X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
        # projection on the first principal component (in the phi space)
        Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
        plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

        plt.subplot(2, 2, 2, aspect='equal')
        plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red",
                    s=20, edgecolor='k')
        plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue",
                    s=20, edgecolor='k')
        plt.title("Projection by PCA")
        plt.xlabel("1st principal component")
        plt.ylabel("2nd component")

        plt.subplot(2, 2, 3, aspect='equal')
        plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
                    s=20, edgecolor='k')
        plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
                    s=20, edgecolor='k')
        plt.title("Projection by KPCA")
        plt.xlabel(r"1st principal component in space induced by $\phi$")
        plt.ylabel("2nd component")

        plt.subplot(2, 2, 4, aspect='equal')
        plt.scatter(X_back[reds, 0], X_back[reds, 1], c="red",
                    s=20, edgecolor='k')
        plt.scatter(X_back[blues, 0], X_back[blues, 1], c="blue",
                    s=20, edgecolor='k')
        plt.title("Original space after inverse transform")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

        plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_decomposition_sparse_coding(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from distutils.version import LooseVersion

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.decomposition import SparseCoder


        def ricker_function(resolution, center, width):
            x = np.linspace(0, resolution - 1, resolution)
            x = ((2 / ((np.sqrt(3 * width) * np.pi ** 1 / 4)))
                * (1 - ((x - center) ** 2 / width ** 2))
                * np.exp((-(x - center) ** 2) / (2 * width ** 2)))
            return x


        def ricker_matrix(width, resolution, n_components):
            centers = np.linspace(0, resolution - 1, n_components)
            D = np.empty((n_components, resolution))
            for i, center in enumerate(centers):
                D[i] = ricker_function(resolution, center, width)
            D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
            return D


        resolution = 1024
        subsampling = 3  # subsampling factor
        width = 100
        n_components = resolution // subsampling

        # Compute a wavelet dictionary
        D_fixed = ricker_matrix(width=width, resolution=resolution,
                                n_components=n_components)
        D_multi = np.r_[tuple(ricker_matrix(width=w, resolution=resolution,
                            n_components=n_components // 5)
                        for w in (10, 50, 100, 500, 1000))]

        # Generate a signal
        y = np.linspace(0, resolution - 1, resolution)
        first_quarter = y < resolution / 4
        y[first_quarter] = 3.
        y[np.logical_not(first_quarter)] = -1.

        # List the different sparse coding methods in the following format:
        # (title, transform_algorithm, transform_alpha, transform_n_nozero_coefs)
        estimators = [('OMP', 'omp', None, 15, 'navy'),
                    ('Lasso', 'lasso_cd', 2, None, 'turquoise'), ]
        lw = 2
        # Avoid FutureWarning about default value change when numpy >= 1.14
        lstsq_rcond = None if LooseVersion(np.__version__) >= '1.14' else -1

        plt.figure(figsize=(13, 6))
        for subplot, (D, title) in enumerate(zip((D_fixed, D_multi),
                                                ('fixed width', 'multiple widths'))):
            plt.subplot(1, 2, subplot + 1)
            plt.title('Sparse coding against %s dictionary' % title)
            plt.plot(y, lw=lw, linestyle='--', label='Original signal')
            # Do a wavelet approximation
            for title, algo, alpha, n_nonzero, color in estimators:
                coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=n_nonzero,
                                    transform_alpha=alpha, transform_algorithm=algo)
                x = coder.transform(y.reshape(1, -1))
                density = len(np.flatnonzero(x))
                x = np.ravel(np.dot(x, D))
                squared_error = np.sum((y - x) ** 2)
                plt.plot(x, color=color, lw=lw,
                        label='%s: %s nonzero coefs,\n%.2f error'
                        % (title, density, squared_error))

            # Soft thresholding debiasing
            coder = SparseCoder(dictionary=D, transform_algorithm='threshold',
                                transform_alpha=20)
            x = coder.transform(y.reshape(1, -1))
            _, idx = np.where(x != 0)
            x[0, idx], _, _, _ = np.linalg.lstsq(D[idx, :].T, y, rcond=lstsq_rcond)
            x = np.ravel(np.dot(x, D))
            squared_error = np.sum((y - x) ** 2)
            plt.plot(x, color='darkorange', lw=lw,
                    label='Thresholding w/ debiasing:\n%d nonzero coefs, %.2f error'
                    % (len(idx), squared_error))
            plt.axis('tight')
            plt.legend(shadow=False, loc='best')
        plt.subplots_adjust(.04, .07, .97, .90, .09, .2)
        plt.show()
    """)
    selenium.run(cmd)