from textwrap import dedent
import pytest

def test_sklearn_nearest_neighbors_regression(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import neighbors

        np.random.seed(0)
        X = np.sort(5 * np.random.rand(40, 1), axis=0)
        T = np.linspace(0, 5, 500)[:, np.newaxis]
        y = np.sin(X).ravel()

        # Add noise to targets
        y[::5] += 1 * (0.5 - np.random.rand(8))

        # #############################################################################
        # Fit regression model
        n_neighbors = 5

        for i, weights in enumerate(['uniform', 'distance']):
            knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            y_ = knn.fit(X, y).predict(T)

            plt.subplot(2, 1, i + 1)
            plt.scatter(X, y, c='k', label='data')
            plt.plot(T, y_, c='g', label='prediction')
            plt.axis('tight')
            plt.legend()
            plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                        weights))

        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_nearest_neighbors_local_outlier_factor(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.neighbors import LocalOutlierFactor

        print(__doc__)

        np.random.seed(42)

        # Generate train data
        X_inliers = 0.3 * np.random.randn(100, 2)
        X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

        # Generate some outliers
        X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
        X = np.r_[X_inliers, X_outliers]

        n_outliers = len(X_outliers)
        ground_truth = np.ones(len(X), dtype=int)
        ground_truth[-n_outliers:] = -1

        # fit the model for outlier detection (default)
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        # use fit_predict to compute the predicted labels of the training samples
        # (when LOF is used for outlier detection, the estimator has no predict,
        # decision_function and score_samples methods).
        y_pred = clf.fit_predict(X)
        n_errors = (y_pred != ground_truth).sum()
        X_scores = clf.negative_outlier_factor_

        plt.title("Local Outlier Factor (LOF)")
        plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
        # plot circles with radius proportional to the outlier scores
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
                    facecolors='none', label='Outlier scores')
        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.xlabel("prediction errors: %d" % (n_errors))
        legend = plt.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_nearest_neighbors_KDE_digits(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.datasets import load_digits
        from sklearn.neighbors import KernelDensity
        from sklearn.decomposition import PCA
        from sklearn.model_selection import GridSearchCV

        # load the data
        digits = load_digits()

        # project the 64-dimensional data to a lower dimension
        pca = PCA(n_components=15, whiten=False)
        data = pca.fit_transform(digits.data)

        # use grid search cross-validation to optimize the bandwidth
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=5)
        grid.fit(data)

        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_

        # sample 44 new points from the data
        new_data = kde.sample(44, random_state=0)
        new_data = pca.inverse_transform(new_data)

        # turn data into a 4x11 grid
        new_data = new_data.reshape((4, 11, -1))
        real_data = digits.data[:44].reshape((4, 11, -1))

        # plot real digits and resampled digits
        fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
        for j in range(11):
            ax[4, j].set_visible(False)
            for i in range(4):
                im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                    cmap=plt.cm.binary, interpolation='nearest')
                im.set_clim(0, 16)
                im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                        cmap=plt.cm.binary, interpolation='nearest')
                im.set_clim(0, 16)

        ax[0, 5].set_title('Selection from the input data')
        ax[5, 5].set_title('"New" digits drawn from the kernel density model')

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_nearest_neighbors_novelty_detection_with_LOF(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        from sklearn.neighbors import LocalOutlierFactor

        print(__doc__)

        np.random.seed(42)

        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
        # Generate normal (not abnormal) training observations
        X = 0.3 * np.random.randn(100, 2)
        X_train = np.r_[X + 2, X - 2]
        # Generate new normal (not abnormal) observations
        X = 0.3 * np.random.randn(20, 2)
        X_test = np.r_[X + 2, X - 2]
        # Generate some abnormal novel observations
        X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

        # fit the model for novelty detection (novelty=True)
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
        clf.fit(X_train)
        # DO NOT use predict, decision_function and score_samples on X_train as this
        # would give wrong results but only on new unseen data (not used in X_train),
        # e.g. X_test, X_outliers or the meshgrid
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)
        n_error_test = y_pred_test[y_pred_test == -1].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

        # plot the learned frontier, the points, and the nearest vectors to the plane
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title("Novelty Detection with LOF")
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

        s = 40
        b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
        b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                        edgecolors='k')
        c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                        edgecolors='k')
        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.legend([a.collections[0], b1, b2, c],
                ["learned frontier", "training observations",
                    "new regular observations", "new abnormal observations"],
                loc="upper left",
                prop=matplotlib.font_manager.FontProperties(size=11))
        plt.xlabel(
            "errors novel regular: %d/40 ; errors novel abnormal: %d/40"
            % (n_error_test, n_error_outliers))
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_nearest_neighbors_KDE_of_species_distribution(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import fetch_species_distributions
        from sklearn.datasets.species_distributions import construct_grids
        from sklearn.neighbors import KernelDensity

        # if basemap is available, we'll use it.
        # otherwise, we'll improvise later...
        try:
            from mpl_toolkits.basemap import Basemap
            basemap = True
        except ImportError:
            basemap = False

        # Get matrices/arrays of species IDs and locations
        data = fetch_species_distributions()
        species_names = ['Bradypus Variegatus', 'Microryzomys Minutus']

        Xtrain = np.vstack([data['train']['dd lat'],
                            data['train']['dd long']]).T
        ytrain = np.array([d.decode('ascii').startswith('micro')
                        for d in data['train']['species']], dtype='int')
        Xtrain *= np.pi / 180.  # Convert lat/long to radians

        # Set up the data grid for the contour plot
        xgrid, ygrid = construct_grids(data)
        X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
        land_reference = data.coverages[6][::5, ::5]
        land_mask = (land_reference > -9999).ravel()

        xy = np.vstack([Y.ravel(), X.ravel()]).T
        xy = xy[land_mask]
        xy *= np.pi / 180.

        # Plot map of South America with distributions of each species
        fig = plt.figure()
        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)

        for i in range(2):
            plt.subplot(1, 2, i + 1)

            # construct a kernel density estimate of the distribution
            print(" - computing KDE in spherical coordinates")
            kde = KernelDensity(bandwidth=0.04, metric='haversine',
                                kernel='gaussian', algorithm='ball_tree')
            kde.fit(Xtrain[ytrain == i])

            # evaluate only on the land: -9999 indicates ocean
            Z = np.full(land_mask.shape[0], -9999, dtype='int')
            Z[land_mask] = np.exp(kde.score_samples(xy))
            Z = Z.reshape(X.shape)

            # plot contours of the density
            levels = np.linspace(0, Z.max(), 25)
            plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)

            if basemap:
                print(" - plot coastlines using basemap")
                m = Basemap(projection='cyl', llcrnrlat=Y.min(),
                            urcrnrlat=Y.max(), llcrnrlon=X.min(),
                            urcrnrlon=X.max(), resolution='c')
                m.drawcoastlines()
                m.drawcountries()
            else:
                print(" - plot coastlines from coverage")
                plt.contour(X, Y, land_reference,
                            levels=[-9998], colors="k",
                            linestyles="solid")
                plt.xticks([])
                plt.yticks([])

            plt.title(species_names[i])

        plt.show()
    """)
    selenium.run(cmd)
