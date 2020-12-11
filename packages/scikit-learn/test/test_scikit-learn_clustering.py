from textwrap import dedent
import pytest

def test_sklearn_clustering_digits_agglomeration(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import datasets, cluster
        from sklearn.feature_extraction.image import grid_to_graph

        digits = datasets.load_digits()
        images = digits.images
        X = np.reshape(images, (len(images), -1))
        connectivity = grid_to_graph(*images[0].shape)

        agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                            n_clusters=32)

        agglo.fit(X)
        X_reduced = agglo.transform(X)

        X_restored = agglo.inverse_transform(X_reduced)
        images_restored = np.reshape(X_restored, images.shape)
        plt.figure(1, figsize=(4, 3.5))
        plt.clf()
        plt.subplots_adjust(left=.01, right=.99, bottom=.01, top=.91)
        for i in range(4):
            plt.subplot(3, 4, i + 1)
            plt.imshow(images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
            if i == 1:
                plt.title('Original data')
            plt.subplot(3, 4, 4 + i + 1)
            plt.imshow(images_restored[i], cmap=plt.cm.gray, vmax=16,
                    interpolation='nearest')
            if i == 1:
                plt.title('Agglomerated data')
            plt.xticks(())
            plt.yticks(())

        plt.subplot(3, 4, 10)
        plt.imshow(np.reshape(agglo.labels_, images[0].shape),
                interpolation='nearest', cmap=plt.cm.nipy_spectral)
        plt.xticks(())
        plt.yticks(())
        plt.title('Labels')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_clustering_face_compress(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import scipy as sp
        import matplotlib.pyplot as plt

        from sklearn import cluster


        try:  # SciPy >= 0.16 have face in misc
            from scipy.misc import face
            face = face(gray=True)
        except ImportError:
            face = sp.face(gray=True)

        n_clusters = 5
        np.random.seed(0)

        X = face.reshape((-1, 1))  # We need an (n_sample, n_feature) array
        k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
        k_means.fit(X)
        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_

        # create an array from labels and values
        face_compressed = np.choose(labels, values)
        face_compressed.shape = face.shape

        vmin = face.min()
        vmax = face.max()

        # original face
        plt.figure(1, figsize=(3, 2.2))
        plt.imshow(face, cmap=plt.cm.gray, vmin=vmin, vmax=256)

        # compressed face
        plt.figure(2, figsize=(3, 2.2))
        plt.imshow(face_compressed, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

        # equal bins face
        regular_values = np.linspace(0, 256, n_clusters + 1)
        regular_labels = np.searchsorted(regular_values, face) - 1
        regular_values = .5 * (regular_values[1:] + regular_values[:-1])  # mean
        regular_face = np.choose(regular_labels.ravel(), regular_values, mode="clip")
        regular_face.shape = face.shape
        plt.figure(3, figsize=(3, 2.2))
        plt.imshow(regular_face, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

        # histogram
        plt.figure(4, figsize=(3, 2.2))
        plt.clf()
        plt.axes([.01, .01, .98, .98])
        plt.hist(X, bins=256, color='.5', edgecolor='.5')
        plt.yticks(())
        plt.xticks(regular_values)
        values = np.sort(values)
        for center_1, center_2 in zip(values[:-1], values[1:]):
            plt.axvline(.5 * (center_1 + center_2), color='b')

        for center_1, center_2 in zip(regular_values[:-1], regular_values[1:]):
            plt.axvline(.5 * (center_1 + center_2), color='b', linestyle='--')

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_clustering_k_means(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        # Though the following import is not directly being used, it is required
        # for 3D projection to work
        from mpl_toolkits.mplot3d import Axes3D

        from sklearn.cluster import KMeans
        from sklearn import datasets

        np.random.seed(5)

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
                    ('k_means_iris_3', KMeans(n_clusters=3)),
                    ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                                    init='random'))]

        fignum = 1
        titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
        for name, est in estimators:
            fig = plt.figure(fignum, figsize=(4, 3))
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
            est.fit(X)
            labels = est.labels_

            ax.scatter(X[:, 3], X[:, 0], X[:, 2],
                    c=labels.astype(np.float), edgecolor='k')

            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_xlabel('Petal width')
            ax.set_ylabel('Sepal length')
            ax.set_zlabel('Petal length')
            ax.set_title(titles[fignum - 1])
            ax.dist = 12
            fignum = fignum + 1

        # Plot the ground truth
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        for name, label in [('Setosa', 0),
                            ('Versicolour', 1),
                            ('Virginica', 2)]:
            ax.text3D(X[y == label, 3].mean(),
                    X[y == label, 0].mean(),
                    X[y == label, 2].mean() + 2, name,
                    horizontalalignment='center',
                    bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title('Ground Truth')
        ax.dist = 12

        fig.show()
    """)
    selenium.run(cmd)

def test_sklearn_clustering_DBSCAN(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np

        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.preprocessing import StandardScaler


        # #############################################################################
        # Generate sample data
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                    random_state=0)

        X = StandardScaler().fit_transform(X)

        # #############################################################################
        # Compute DBSCAN
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))

        # #############################################################################
        # Plot result
        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_clustering_hierarchical_linkage_comparison(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import time
        import warnings

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import cluster, datasets
        from sklearn.preprocessing import StandardScaler
        from itertools import cycle, islice

        np.random.seed(0)
        n_samples = 1500
        noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                            noise=.05)
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
        blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
        no_structure = np.random.rand(n_samples, 2), None

        # Anisotropicly distributed data
        random_state = 170
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        aniso = (X_aniso, y)

        # blobs with varied variances
        varied = datasets.make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)

        plt.figure(figsize=(9 * 1.3 + 2, 14.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)

        plot_num = 1

        default_base = {'n_neighbors': 10,
                        'n_clusters': 3}

        datasets = [
            (noisy_circles, {'n_clusters': 2}),
            (noisy_moons, {'n_clusters': 2}),
            (varied, {'n_neighbors': 2}),
            (aniso, {'n_neighbors': 2}),
            (blobs, {}),
            (no_structure, {})]

        for i_dataset, (dataset, algo_params) in enumerate(datasets):
            # update parameters with dataset-specific values
            params = default_base.copy()
            params.update(algo_params)

            X, y = dataset

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # ============
            # Create cluster objects
            # ============
            ward = cluster.AgglomerativeClustering(
                n_clusters=params['n_clusters'], linkage='ward')
            complete = cluster.AgglomerativeClustering(
                n_clusters=params['n_clusters'], linkage='complete')
            average = cluster.AgglomerativeClustering(
                n_clusters=params['n_clusters'], linkage='average')
            single = cluster.AgglomerativeClustering(
                n_clusters=params['n_clusters'], linkage='single')

            clustering_algorithms = (
                ('Single Linkage', single),
                ('Average Linkage', average),
                ('Complete Linkage', complete),
                ('Ward Linkage', ward),
            )

            for name, algorithm in clustering_algorithms:
                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning)
                    algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)

                plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#999999', '#e41a1c', '#dede00']),
                                            int(max(y_pred) + 1))))
                plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                        transform=plt.gca().transAxes, size=15,
                        horizontalalignment='right')
                plot_num += 1

        plt.show()
    """)
    selenium.run(cmd)