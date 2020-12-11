from textwrap import dedent
import pytest

def test_sklearn_manifold_swiss_roll_reduction_with_LLE(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt

        # This import is needed to modify the way figure behaves
        from mpl_toolkits.mplot3d import Axes3D
        Axes3D

        #----------------------------------------------------------------------
        # Locally linear embedding of the swiss roll

        from sklearn import manifold, datasets
        X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500)

        print("Computing LLE embedding")
        X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12,
                                                    n_components=2)
        print("Done. Reconstruction error: %g" % err)

        #----------------------------------------------------------------------
        # Plot result

        fig = plt.figure()

        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

        ax.set_title("Original data")
        ax = fig.add_subplot(212)
        ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.axis('tight')
        plt.xticks([]), plt.yticks([])
        plt.title('Projected data')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_manifold_multidimensional_scaling(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np

        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection

        from sklearn import manifold
        from sklearn.metrics import euclidean_distances
        from sklearn.decomposition import PCA

        n_samples = 20
        seed = np.random.RandomState(seed=3)
        X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
        X_true = X_true.reshape((n_samples, 2))
        # Center the data
        X_true -= X_true.mean()

        similarities = euclidean_distances(X_true)

        # Add noise to the similarities
        noise = np.random.rand(n_samples, n_samples)
        noise = noise + noise.T
        noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
        similarities += noise

        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                        dissimilarity="precomputed", n_jobs=1)
        pos = mds.fit(similarities).embedding_

        nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                            dissimilarity="precomputed", random_state=seed, n_jobs=1,
                            n_init=1)
        npos = nmds.fit_transform(similarities, init=pos)

        # Rescale the data
        pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())
        npos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((npos ** 2).sum())

        # Rotate the data
        clf = PCA(n_components=2)
        X_true = clf.fit_transform(X_true)

        pos = clf.fit_transform(pos)

        npos = clf.fit_transform(npos)

        fig = plt.figure(1)
        ax = plt.axes([0., 0., 1., 1.])

        s = 100
        plt.scatter(X_true[:, 0], X_true[:, 1], color='navy', s=s, lw=0,
                    label='True Position')
        plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
        plt.scatter(npos[:, 0], npos[:, 1], color='darkorange', s=s, lw=0, label='NMDS')
        plt.legend(scatterpoints=1, loc='best', shadow=False)

        similarities = similarities.max() / similarities * 100
        similarities[np.isinf(similarities)] = 0

        # Plot the edges
        start_idx, end_idx = np.where(pos)
        # a sequence of (*line0*, *line1*, *line2*), where::
        #            linen = (x0, y0), (x1, y1), ... (xm, ym)
        segments = [[X_true[i, :], X_true[j, :]]
                    for i in range(len(pos)) for j in range(len(pos))]
        values = np.abs(similarities)
        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.Blues,
                            norm=plt.Normalize(0, values.max()))
        lc.set_array(similarities.flatten())
        lc.set_linewidths(np.full(len(segments), 0.5))
        ax.add_collection(lc)

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_manifold_learning_on_severed_sphere(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from time import time

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.ticker import NullFormatter

        from sklearn import manifold
        from sklearn.utils import check_random_state

        # Next line to silence pyflakes.
        Axes3D

        # Variables for manifold learning.
        n_neighbors = 10
        n_samples = 1000

        # Create our sphere.
        random_state = check_random_state(0)
        p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
        t = random_state.rand(n_samples) * np.pi

        # Sever the poles from the sphere.
        indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
        colors = p[indices]
        x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
            np.sin(t[indices]) * np.sin(p[indices]), \
            np.cos(t[indices])

        # Plot our dataset.
        fig = plt.figure(figsize=(15, 8))
        plt.suptitle("Manifold Learning with %i points, %i neighbors"
                    % (1000, n_neighbors), fontsize=14)

        ax = fig.add_subplot(251, projection='3d')
        ax.scatter(x, y, z, c=p[indices], cmap=plt.cm.rainbow)
        ax.view_init(40, -10)

        sphere_data = np.array([x, y, z]).T

        # Perform Locally Linear Embedding Manifold learning
        methods = ['standard', 'ltsa', 'hessian', 'modified']
        labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

        for i, method in enumerate(methods):
            t0 = time()
            trans_data = manifold\
                .LocallyLinearEmbedding(n_neighbors, 2,
                                        method=method).fit_transform(sphere_data).T
            t1 = time()
            print("%s: %.2g sec" % (methods[i], t1 - t0))

            ax = fig.add_subplot(252 + i)
            plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
            plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')

        # Perform Isomap Manifold learning.
        t0 = time()
        trans_data = manifold.Isomap(n_neighbors, n_components=2)\
            .fit_transform(sphere_data).T
        t1 = time()
        print("%s: %.2g sec" % ('ISO', t1 - t0))

        ax = fig.add_subplot(257)
        plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
        plt.title("%s (%.2g sec)" % ('Isomap', t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

        # Perform Multi-dimensional scaling.
        t0 = time()
        mds = manifold.MDS(2, max_iter=100, n_init=1)
        trans_data = mds.fit_transform(sphere_data).T
        t1 = time()
        print("MDS: %.2g sec" % (t1 - t0))

        ax = fig.add_subplot(258)
        plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
        plt.title("MDS (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

        # Perform Spectral Embedding.
        t0 = time()
        se = manifold.SpectralEmbedding(n_components=2,
                                        n_neighbors=n_neighbors)
        trans_data = se.fit_transform(sphere_data).T
        t1 = time()
        print("Spectral Embedding: %.2g sec" % (t1 - t0))

        ax = fig.add_subplot(259)
        plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
        plt.title("Spectral Embedding (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

        # Perform t-distributed stochastic neighbor embedding.
        t0 = time()
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        trans_data = tsne.fit_transform(sphere_data).T
        t1 = time()
        print("t-SNE: %.2g sec" % (t1 - t0))

        ax = fig.add_subplot(2, 5, 10)
        plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
        plt.title("t-SNE (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

        plt.show()
    """)
    selenium.run(cmd)