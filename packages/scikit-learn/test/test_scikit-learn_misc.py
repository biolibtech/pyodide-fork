from textwrap import dedent
import pytest


def test_sklearn_misc_imputation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.datasets import load_diabetes
        from sklearn.datasets import load_boston
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import make_pipeline, make_union
        from sklearn.impute import SimpleImputer, MissingIndicator
        from sklearn.model_selection import cross_val_score

        rng = np.random.RandomState(0)


        def get_results(dataset):
            X_full, y_full = dataset.data, dataset.target
            n_samples = X_full.shape[0]
            n_features = X_full.shape[1]

            # Estimate the score on the entire dataset, with no missing values
            estimator = RandomForestRegressor(random_state=0, n_estimators=100)
            full_scores = cross_val_score(estimator, X_full, y_full,
                                        scoring='neg_mean_squared_error', cv=5)

            # Add missing values in 75% of the lines
            missing_rate = 0.75
            n_missing_samples = int(np.floor(n_samples * missing_rate))
            missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                                dtype=np.bool),
                                        np.ones(n_missing_samples,
                                                dtype=np.bool)))
            rng.shuffle(missing_samples)
            missing_features = rng.randint(0, n_features, n_missing_samples)

            # Estimate the score after replacing missing values by 0
            X_missing = X_full.copy()
            X_missing[np.where(missing_samples)[0], missing_features] = 0
            y_missing = y_full.copy()
            estimator = RandomForestRegressor(random_state=0, n_estimators=100)
            zero_impute_scores = cross_val_score(estimator, X_missing, y_missing,
                                                scoring='neg_mean_squared_error',
                                                cv=5)

            # Estimate the score after imputation (mean strategy) of the missing values
            X_missing = X_full.copy()
            X_missing[np.where(missing_samples)[0], missing_features] = 0
            y_missing = y_full.copy()
            estimator = make_pipeline(
                make_union(SimpleImputer(missing_values=0, strategy="mean"),
                        MissingIndicator(missing_values=0)),
                RandomForestRegressor(random_state=0, n_estimators=100))
            mean_impute_scores = cross_val_score(estimator, X_missing, y_missing,
                                                scoring='neg_mean_squared_error',
                                                cv=5)


            return ((full_scores.mean(), full_scores.std()),
                    (zero_impute_scores.mean(), zero_impute_scores.std()),
                    (mean_impute_scores.mean(), mean_impute_scores.std()))


        results_diabetes = np.array(get_results(load_diabetes()))
        mses_diabetes = results_diabetes[:, 0] * -1
        stds_diabetes = results_diabetes[:, 1]

        results_boston = np.array(get_results(load_boston()))
        mses_boston = results_boston[:, 0] * -1
        stds_boston = results_boston[:, 1]

        n_bars = len(mses_diabetes)
        xval = np.arange(n_bars)

        x_labels = ['Full data',
                    'Zero imputation',
                    'Mean Imputation']
        colors = ['r', 'g', 'b', 'orange']

        # plot diabetes results
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(121)
        for j in xval:
            ax1.barh(j, mses_diabetes[j], xerr=stds_diabetes[j],
                    color=colors[j], alpha=0.6, align='center')

        ax1.set_title('Imputation Techniques with Diabetes Data')
        ax1.set_xlim(left=np.min(mses_diabetes) * 0.9,
                    right=np.max(mses_diabetes) * 1.1)
        ax1.set_yticks(xval)
        ax1.set_xlabel('MSE')
        ax1.invert_yaxis()
        ax1.set_yticklabels(x_labels)

        # plot boston results
        ax2 = plt.subplot(122)
        for j in xval:
            ax2.barh(j, mses_boston[j], xerr=stds_boston[j],
                    color=colors[j], alpha=0.6, align='center')

        ax2.set_title('Imputation Techniques with Boston Data')
        ax2.set_yticks(xval)
        ax2.set_xlabel('MSE')
        ax2.invert_yaxis()
        ax2.set_yticklabels([''] * n_bars)
        assert(len(results_diabetes) > 0)

        plt.show()
    """)
    selenium.run(cmd)


def test_sklearn_misc_multi_label_classification(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.datasets import make_multilabel_classification
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
        from sklearn.decomposition import PCA
        from sklearn.cross_decomposition import CCA


        def plot_hyperplane(clf, min_x, max_x, linestyle, label):
            # get the separating hyperplane
            w = clf.coef_[0]
            a = -w[0] / w[1]
            xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
            yy = a * xx - (clf.intercept_[0]) / w[1]
            plt.plot(xx, yy, linestyle, label=label)


        def plot_subfigure(X, Y, subplot, title, transform):
            if transform == "pca":
                X = PCA(n_components=2).fit_transform(X)
            elif transform == "cca":
                X = CCA(n_components=2).fit(X, Y).transform(X)
            else:
                raise ValueError

            min_x = np.min(X[:, 0])
            max_x = np.max(X[:, 0])

            min_y = np.min(X[:, 1])
            max_y = np.max(X[:, 1])

            classif = OneVsRestClassifier(SVC(kernel='linear'))
            classif.fit(X, Y)

            plt.subplot(2, 2, subplot)
            plt.title(title)

            zero_class = np.where(Y[:, 0])
            one_class = np.where(Y[:, 1])
            plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
            plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                        facecolors='none', linewidths=2, label='Class 1')
            plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                        facecolors='none', linewidths=2, label='Class 2')

            plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                            'Boundary\nfor class 1')
            plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                            'Boundary\nfor class 2')
            plt.xticks(())
            plt.yticks(())

            plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
            plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
            if subplot == 2:
                plt.xlabel('First principal component')
                plt.ylabel('Second principal component')
                plt.legend(loc="upper left")


        plt.figure(figsize=(8, 6))

        X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                            allow_unlabeled=True,
                                            random_state=1)

        plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
        plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

        X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                            allow_unlabeled=False,
                                            random_state=1)

        plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
        plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

        plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
        plt.show()
    """)
    selenium.run(cmd)


def test_skimage_transform_isotonic_regression(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        from sklearn.linear_model import LinearRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.utils import check_random_state

        n = 100
        x = np.arange(n)
        rs = check_random_state(0)
        y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))

        # #############################################################################
        # Fit IsotonicRegression and LinearRegression models

        ir = IsotonicRegression()

        y_ = ir.fit_transform(x, y)

        lr = LinearRegression()
        lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

        # #############################################################################
        # Plot result

        segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
        lc = LineCollection(segments, zorder=0)
        lc.set_array(np.ones(len(y)))
        lc.set_linewidths(np.full(n, 0.5))

        fig = plt.figure()
        plt.plot(x, y, 'r.', markersize=12)
        plt.plot(x, y_, 'g.-', markersize=12)
        plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
        plt.gca().add_collection(lc)
        plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
        plt.title('Isotonic regression')
        plt.show()
    """)
    selenium.run(cmd)


def test_sklearn_misc_anomaly_detection_comparison(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import time

        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt

        from sklearn import svm
        from sklearn.datasets import make_moons, make_blobs
        from sklearn.covariance import EllipticEnvelope
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        print(__doc__)

        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

        # Example settings
        n_samples = 300
        outliers_fraction = 0.15
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers

        # define outlier/anomaly detection methods to be compared
        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
            ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                            gamma=0.1)),
            ("Isolation Forest", IsolationForest(behaviour='new',
                                                contamination=outliers_fraction,
                                                random_state=42)),
            ("Local Outlier Factor", LocalOutlierFactor(
                n_neighbors=35, contamination=outliers_fraction))]

        # Define datasets
        blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
        datasets = [
            make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
                    **blobs_params)[0],
            make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
                    **blobs_params)[0],
            make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
                    **blobs_params)[0],
            4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
                np.array([0.5, 0.25])),
            14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

        # Compare given classifiers under given settings
        xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                            np.linspace(-7, 7, 150))

        plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)

        plot_num = 1
        rng = np.random.RandomState(42)

        for i_dataset, X in enumerate(datasets):
            # Add outliers
            X = np.concatenate([X, rng.uniform(low=-6, high=6,
                            size=(n_outliers, 2))], axis=0)

            for name, algorithm in anomaly_algorithms:
                t0 = time.time()
                algorithm.fit(X)
                t1 = time.time()
                plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)

                # fit the data and tag outliers
                if name == "Local Outlier Factor":
                    y_pred = algorithm.fit_predict(X)
                else:
                    y_pred = algorithm.fit(X).predict(X)

                # plot the levels lines and the points
                if name != "Local Outlier Factor":  # LOF does not implement predict
                    Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

                colors = np.array(['#377eb8', '#ff7f00'])
                plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

                plt.xlim(-7, 7)
                plt.ylim(-7, 7)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                        transform=plt.gca().transAxes, size=15,
                        horizontalalignment='right')
                plot_num += 1

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_misc_johnson_lindenstrauss_bond(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import sys
        from time import time
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        from distutils.version import LooseVersion
        from sklearn.random_projection import johnson_lindenstrauss_min_dim
        from sklearn.random_projection import SparseRandomProjection
        from sklearn.datasets import fetch_20newsgroups_vectorized
        from sklearn.datasets import load_digits
        from sklearn.metrics.pairwise import euclidean_distances

        # `normed` is being deprecated in favor of `density` in histograms
        if LooseVersion(matplotlib.__version__) >= '2.1':
            density_param = {'density': True}
        else:
            density_param = {'normed': True}

        # Part 1: plot the theoretical dependency between n_components_min and
        # n_samples

        # range of admissible distortions
        eps_range = np.linspace(0.1, 0.99, 5)
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))

        # range of number of samples (observation) to embed
        n_samples_range = np.logspace(1, 9, 9)

        plt.figure()
        for eps, color in zip(eps_range, colors):
            min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
            plt.loglog(n_samples_range, min_n_components, color=color)

        plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="lower right")
        plt.xlabel("Number of observations to eps-embed")
        plt.ylabel("Minimum number of dimensions")
        plt.title("Johnson-Lindenstrauss bounds:\nn_samples vs n_components")

        # range of admissible distortions
        eps_range = np.linspace(0.01, 0.99, 100)

        # range of number of samples (observation) to embed
        n_samples_range = np.logspace(2, 6, 5)
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))

        plt.figure()
        for n_samples, color in zip(n_samples_range, colors):
            min_n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps_range)
            plt.semilogy(eps_range, min_n_components, color=color)

        plt.legend(["n_samples = %d" % n for n in n_samples_range], loc="upper right")
        plt.xlabel("Distortion eps")
        plt.ylabel("Minimum number of dimensions")
        plt.title("Johnson-Lindenstrauss bounds:\nn_components vs eps")

        # Part 2: perform sparse random projection of some digits images which are
        # quite low dimensional and dense or documents of the 20 newsgroups dataset
        # which is both high dimensional and sparse

        if '--twenty-newsgroups' in sys.argv:
            # Need an internet connection hence not enabled by default
            data = fetch_20newsgroups_vectorized().data[:500]
        else:
            data = load_digits().data[:500]

        n_samples, n_features = data.shape
        print("Embedding %d samples with dim %d using various random projections"
            % (n_samples, n_features))

        n_components_range = np.array([300, 1000, 10000])
        dists = euclidean_distances(data, squared=True).ravel()

        # select only non-identical samples pairs
        nonzero = dists != 0
        dists = dists[nonzero]

        for n_components in n_components_range:
            t0 = time()
            rp = SparseRandomProjection(n_components=n_components)
            projected_data = rp.fit_transform(data)
            print("Projected %d samples from %d to %d in %0.3fs"
                % (n_samples, n_features, n_components, time() - t0))
            if hasattr(rp, 'components_'):
                n_bytes = rp.components_.data.nbytes
                n_bytes += rp.components_.indices.nbytes
                print("Random matrix with size: %0.3fMB" % (n_bytes / 1e6))

            projected_dists = euclidean_distances(
                projected_data, squared=True).ravel()[nonzero]

            plt.figure()
            plt.hexbin(dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu)
            plt.xlabel("Pairwise squared distances in original space")
            plt.ylabel("Pairwise squared distances in projected space")
            plt.title("Pairwise distances distribution for n_components=%d" %
                    n_components)
            cb = plt.colorbar()
            cb.set_label('Sample pairs counts')

            rates = projected_dists / dists
            print("Mean distances rate: %0.2f (%0.2f)"
                % (np.mean(rates), np.std(rates)))

            plt.figure()
            plt.hist(rates, bins=50, range=(0., 2.), edgecolor='k', **density_param)
            plt.xlabel("Squared distances rate: projected / original")
            plt.ylabel("Distribution of samples pairs")
            plt.title("Histogram of pairwise distance rates for n_components=%d" %
                    n_components)

            # TODO: compute the expected value of eps and add them to the previous plot
            # as vertical lines / region

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_misc_kernel_ridge_regression_comparison(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from __future__ import division
        import time

        import numpy as np

        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import learning_curve
        from sklearn.kernel_ridge import KernelRidge
        import matplotlib.pyplot as plt

        rng = np.random.RandomState(0)

        # #############################################################################
        # Generate sample data
        X = 5 * rng.rand(10000, 1)
        y = np.sin(X).ravel()

        # Add noise to targets
        y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

        X_plot = np.linspace(0, 5, 100000)[:, None]

        # #############################################################################
        # Fit regression model
        train_size = 100
        svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                        param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                    "gamma": np.logspace(-2, 2, 5)})

        kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                    "gamma": np.logspace(-2, 2, 5)})

        t0 = time.time()
        svr.fit(X[:train_size], y[:train_size])
        svr_fit = time.time() - t0
        print("SVR complexity and bandwidth selected and model fitted in %.3f s"
            % svr_fit)

        t0 = time.time()
        kr.fit(X[:train_size], y[:train_size])
        kr_fit = time.time() - t0
        print("KRR complexity and bandwidth selected and model fitted in %.3f s"
            % kr_fit)

        sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
        print("Support vector ratio: %.3f" % sv_ratio)

        t0 = time.time()
        y_svr = svr.predict(X_plot)
        svr_predict = time.time() - t0
        print("SVR prediction for %d inputs in %.3f s"
            % (X_plot.shape[0], svr_predict))

        t0 = time.time()
        y_kr = kr.predict(X_plot)
        kr_predict = time.time() - t0
        print("KRR prediction for %d inputs in %.3f s"
            % (X_plot.shape[0], kr_predict))


        # #############################################################################
        # Look at the results
        sv_ind = svr.best_estimator_.support_
        plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
                    zorder=2, edgecolors=(0, 0, 0))
        plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1,
                    edgecolors=(0, 0, 0))
        plt.plot(X_plot, y_svr, c='r',
                label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
        plt.plot(X_plot, y_kr, c='g',
                label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
        plt.xlabel('data')
        plt.ylabel('target')
        plt.title('SVR versus Kernel Ridge')
        plt.legend()

        # Visualize training and prediction time
        plt.figure()

        # Generate sample data
        X = 5 * rng.rand(10000, 1)
        y = np.sin(X).ravel()
        y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
        sizes = np.logspace(1, 4, 7).astype(np.int)
        for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
                                                gamma=10),
                                "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
            train_time = []
            test_time = []
            for train_test_size in sizes:
                t0 = time.time()
                estimator.fit(X[:train_test_size], y[:train_test_size])
                train_time.append(time.time() - t0)

                t0 = time.time()
                estimator.predict(X_plot[:1000])
                test_time.append(time.time() - t0)

            plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
                    label="%s (train)" % name)
            plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
                    label="%s (test)" % name)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Train size")
        plt.ylabel("Time (seconds)")
        plt.title('Execution Time')
        plt.legend(loc="best")

        # Visualize learning curves
        plt.figure()

        svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
        kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
        train_sizes, train_scores_svr, test_scores_svr = \
            learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                        scoring="neg_mean_squared_error", cv=10)
        train_sizes_abs, train_scores_kr, test_scores_kr = \
            learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                        scoring="neg_mean_squared_error", cv=10)

        plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
                label="SVR")
        plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g",
                label="KRR")
        plt.xlabel("Train size")
        plt.ylabel("Mean Squared Error")
        plt.title('Learning curves')
        plt.legend(loc="best")

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_misc_kernel_approximation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        from time import time

        # Import datasets, classifiers and performance metrics
        from sklearn import datasets, svm, pipeline
        from sklearn.kernel_approximation import (RBFSampler,
                                                Nystroem)
        from sklearn.decomposition import PCA

        # The digits dataset
        digits = datasets.load_digits(n_class=9)

        # To apply an classifier on this data, we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        n_samples = len(digits.data)
        data = digits.data / 16.
        data -= data.mean(axis=0)

        # We learn the digits on the first half of the digits
        data_train, targets_train = (data[:n_samples // 2],
                                    digits.target[:n_samples // 2])


        # Now predict the value of the digit on the second half:
        data_test, targets_test = (data[n_samples // 2:],
                                digits.target[n_samples // 2:])
        # data_test = scaler.transform(data_test)

        # Create a classifier: a support vector classifier
        kernel_svm = svm.SVC(gamma=.2)
        linear_svm = svm.LinearSVC()

        # create pipeline from kernel approximation
        # and linear svm
        feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
        feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
        fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                                ("svm", svm.LinearSVC())])

        nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                                ("svm", svm.LinearSVC())])

        # fit and predict using linear and kernel svm:

        kernel_svm_time = time()
        kernel_svm.fit(data_train, targets_train)
        kernel_svm_score = kernel_svm.score(data_test, targets_test)
        kernel_svm_time = time() - kernel_svm_time

        linear_svm_time = time()
        linear_svm.fit(data_train, targets_train)
        linear_svm_score = linear_svm.score(data_test, targets_test)
        linear_svm_time = time() - linear_svm_time

        sample_sizes = 30 * np.arange(1, 10)
        fourier_scores = []
        nystroem_scores = []
        fourier_times = []
        nystroem_times = []

        for D in sample_sizes:
            fourier_approx_svm.set_params(feature_map__n_components=D)
            nystroem_approx_svm.set_params(feature_map__n_components=D)
            start = time()
            nystroem_approx_svm.fit(data_train, targets_train)
            nystroem_times.append(time() - start)

            start = time()
            fourier_approx_svm.fit(data_train, targets_train)
            fourier_times.append(time() - start)

            fourier_score = fourier_approx_svm.score(data_test, targets_test)
            nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
            nystroem_scores.append(nystroem_score)
            fourier_scores.append(fourier_score)

        # plot the results:
        plt.figure(figsize=(8, 8))
        accuracy = plt.subplot(211)
        # second y axis for timeings
        timescale = plt.subplot(212)

        accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
        timescale.plot(sample_sizes, nystroem_times, '--',
                    label='Nystroem approx. kernel')

        accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
        timescale.plot(sample_sizes, fourier_times, '--',
                    label='Fourier approx. kernel')

        # horizontal lines for exact rbf and linear kernels:
        accuracy.plot([sample_sizes[0], sample_sizes[-1]],
                    [linear_svm_score, linear_svm_score], label="linear svm")
        timescale.plot([sample_sizes[0], sample_sizes[-1]],
                    [linear_svm_time, linear_svm_time], '--', label='linear svm')

        accuracy.plot([sample_sizes[0], sample_sizes[-1]],
                    [kernel_svm_score, kernel_svm_score], label="rbf svm")
        timescale.plot([sample_sizes[0], sample_sizes[-1]],
                    [kernel_svm_time, kernel_svm_time], '--', label='rbf svm')

        # vertical line for dataset dimensionality = 64
        accuracy.plot([64, 64], [0.7, 1], label="n_features")

        # legends and labels
        accuracy.set_title("Classification accuracy")
        timescale.set_title("Training times")
        accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
        accuracy.set_xticks(())
        accuracy.set_ylim(np.min(fourier_scores), 1)
        timescale.set_xlabel("Sampling steps = transformed feature dimension")
        accuracy.set_ylabel("Classification accuracy")
        timescale.set_ylabel("Training time in seconds")
        accuracy.legend(loc='best')
        timescale.legend(loc='best')

        # visualize the decision surface, projected down to the first
        # two principal components of the dataset
        pca = PCA(n_components=8).fit(data_train)

        X = pca.transform(data_train)

        # Generate grid along first two principal components
        multiples = np.arange(-2, 2, 0.1)
        # steps along first component
        first = multiples[:, np.newaxis] * pca.components_[0, :]
        # steps along second component
        second = multiples[:, np.newaxis] * pca.components_[1, :]
        # combine
        grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
        flat_grid = grid.reshape(-1, data.shape[1])

        # title for the plots
        titles = ['SVC with rbf kernel',
                'SVC (linear kernel)\n with Fourier rbf feature map\n'
                'n_components=100',
                'SVC (linear kernel)\n with Nystroem rbf feature map\n'
                'n_components=100']

        plt.tight_layout()
        plt.figure(figsize=(12, 5))

        # predict and plot
        for i, clf in enumerate((kernel_svm, nystroem_approx_svm,
                                fourier_approx_svm)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(1, 3, i + 1)
            Z = clf.predict(flat_grid)

            # Put the result into a color plot
            Z = Z.reshape(grid.shape[:-1])
            plt.contourf(multiples, multiples, Z, cmap=plt.cm.Paired)
            plt.axis('off')

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=targets_train, cmap=plt.cm.Paired,
                        edgecolors=(0, 0, 0))

            plt.title(titles[i])
        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)