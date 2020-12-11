
from textwrap import dedent
import pytest

def test_sklearn_covariance_lw_vs_oas(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.linalg import toeplitz, cholesky

        from sklearn.covariance import LedoitWolf, OAS

        np.random.seed(0)
        n_features = 100
        # simulation covariance matrix (AR(1) process)
        r = 0.1
        real_cov = toeplitz(r ** np.arange(n_features))
        coloring_matrix = cholesky(real_cov)

        n_samples_range = np.arange(6, 31, 1)
        repeat = 100
        lw_mse = np.zeros((n_samples_range.size, repeat))
        oa_mse = np.zeros((n_samples_range.size, repeat))
        lw_shrinkage = np.zeros((n_samples_range.size, repeat))
        oa_shrinkage = np.zeros((n_samples_range.size, repeat))
        for i, n_samples in enumerate(n_samples_range):
            for j in range(repeat):
                X = np.dot(
                    np.random.normal(size=(n_samples, n_features)), coloring_matrix.T)

                lw = LedoitWolf(store_precision=False, assume_centered=True)
                lw.fit(X)
                lw_mse[i, j] = lw.error_norm(real_cov, scaling=False)
                lw_shrinkage[i, j] = lw.shrinkage_

                oa = OAS(store_precision=False, assume_centered=True)
                oa.fit(X)
                oa_mse[i, j] = oa.error_norm(real_cov, scaling=False)
                oa_shrinkage[i, j] = oa.shrinkage_

        # plot MSE
        plt.subplot(2, 1, 1)
        plt.errorbar(n_samples_range, lw_mse.mean(1), yerr=lw_mse.std(1),
                    label='Ledoit-Wolf', color='navy', lw=2)
        plt.errorbar(n_samples_range, oa_mse.mean(1), yerr=oa_mse.std(1),
                    label='OAS', color='darkorange', lw=2)
        plt.ylabel("Squared error")
        plt.legend(loc="upper right")
        plt.title("Comparison of covariance estimators")
        plt.xlim(5, 31)

        # plot shrinkage coefficient
        plt.subplot(2, 1, 2)
        plt.errorbar(n_samples_range, lw_shrinkage.mean(1), yerr=lw_shrinkage.std(1),
                    label='Ledoit-Wolf', color='navy', lw=2)
        plt.errorbar(n_samples_range, oa_shrinkage.mean(1), yerr=oa_shrinkage.std(1),
                    label='OAS', color='darkorange', lw=2)
        plt.xlabel("n_samples")
        plt.ylabel("Shrinkage")
        plt.legend(loc="lower right")
        plt.ylim(plt.ylim()[0], 1. + (plt.ylim()[1] - plt.ylim()[0]) / 10.)
        plt.xlim(5, 31)

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_covariance_sparse_inverse_cov(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from scipy import linalg
        from sklearn.datasets import make_sparse_spd_matrix
        from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
        import matplotlib.pyplot as plt

        # #############################################################################
        # Generate the data
        n_samples = 60
        n_features = 20

        prng = np.random.RandomState(1)
        prec = make_sparse_spd_matrix(n_features, alpha=.98,
                                    smallest_coef=.4,
                                    largest_coef=.7,
                                    random_state=prng)
        cov = linalg.inv(prec)
        d = np.sqrt(np.diag(cov))
        cov /= d
        cov /= d[:, np.newaxis]
        prec *= d
        prec *= d[:, np.newaxis]
        X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)

        # #############################################################################
        # Estimate the covariance
        emp_cov = np.dot(X.T, X) / n_samples

        model = GraphicalLassoCV(cv=5)
        model.fit(X)
        cov_ = model.covariance_
        prec_ = model.precision_

        lw_cov_, _ = ledoit_wolf(X)
        lw_prec_ = linalg.inv(lw_cov_)

        # #############################################################################
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.02, right=0.98)

        # plot the covariances
        covs = [('Empirical', emp_cov), ('Ledoit-Wolf', lw_cov_),
                ('GraphicalLassoCV', cov_), ('True', cov)]
        vmax = cov_.max()
        for i, (name, this_cov) in enumerate(covs):
            plt.subplot(2, 4, i + 1)
            plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
                    cmap=plt.cm.RdBu_r)
            plt.xticks(())
            plt.yticks(())
            plt.title('%s covariance' % name)


        # plot the precisions
        precs = [('Empirical', linalg.inv(emp_cov)), ('Ledoit-Wolf', lw_prec_),
                ('GraphicalLasso', prec_), ('True', prec)]
        vmax = .9 * prec_.max()
        for i, (name, this_prec) in enumerate(precs):
            ax = plt.subplot(2, 4, i + 5)
            plt.imshow(np.ma.masked_equal(this_prec, 0),
                    interpolation='nearest', vmin=-vmax, vmax=vmax,
                    cmap=plt.cm.RdBu_r)
            plt.xticks(())
            plt.yticks(())
            plt.title('%s precision' % name)
            if hasattr(ax, 'set_facecolor'):
                ax.set_facecolor('.7')
            else:
                ax.set_axis_bgcolor('.7')

        # plot the model selection metric
        plt.figure(figsize=(4, 3))
        plt.axes([.2, .15, .75, .7])
        plt.plot(model.cv_alphas_, np.mean(model.grid_scores_, axis=1), 'o-')
        plt.axvline(model.alpha_, color='.5')
        plt.title('Model selection')
        plt.ylabel('Cross-validation score')
        plt.xlabel('alpha')

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_covariance_mahalanobis_distance(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.covariance import EmpiricalCovariance, MinCovDet

        n_samples = 125
        n_outliers = 25
        n_features = 2

        # generate data
        gen_cov = np.eye(n_features)
        gen_cov[0, 0] = 2.
        X = np.dot(np.random.randn(n_samples, n_features), gen_cov)
        # add some outliers
        outliers_cov = np.eye(n_features)
        outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.
        X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)

        # fit a Minimum Covariance Determinant (MCD) robust estimator to data
        robust_cov = MinCovDet().fit(X)

        # compare estimators learnt from the full data set with true parameters
        emp_cov = EmpiricalCovariance().fit(X)

        # #############################################################################
        # Display results
        fig = plt.figure()
        plt.subplots_adjust(hspace=-.1, wspace=.4, top=.95, bottom=.05)

        # Show data set
        subfig1 = plt.subplot(3, 1, 1)
        inlier_plot = subfig1.scatter(X[:, 0], X[:, 1],
                                    color='black', label='inliers')
        outlier_plot = subfig1.scatter(X[:, 0][-n_outliers:], X[:, 1][-n_outliers:],
                                    color='red', label='outliers')
        subfig1.set_xlim(subfig1.get_xlim()[0], 11.)
        subfig1.set_title("Mahalanobis distances of a contaminated data set:")

        # Show contours of the distance functions
        xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                            np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
        zz = np.c_[xx.ravel(), yy.ravel()]

        mahal_emp_cov = emp_cov.mahalanobis(zz)
        mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
        emp_cov_contour = subfig1.contour(xx, yy, np.sqrt(mahal_emp_cov),
                                        cmap=plt.cm.PuBu_r,
                                        linestyles='dashed')

        mahal_robust_cov = robust_cov.mahalanobis(zz)
        mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
        robust_contour = subfig1.contour(xx, yy, np.sqrt(mahal_robust_cov),
                                        cmap=plt.cm.YlOrBr_r, linestyles='dotted')

        subfig1.legend([emp_cov_contour.collections[1], robust_contour.collections[1],
                        inlier_plot, outlier_plot],
                    ['MLE dist', 'robust dist', 'inliers', 'outliers'],
                    loc="upper right", borderaxespad=0)
        plt.xticks(())
        plt.yticks(())

        # Plot the scores for each point
        emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
        subfig2 = plt.subplot(2, 2, 3)
        subfig2.boxplot([emp_mahal[:-n_outliers], emp_mahal[-n_outliers:]], widths=.25)
        subfig2.plot(np.full(n_samples - n_outliers, 1.26),
                    emp_mahal[:-n_outliers], '+k', markeredgewidth=1)
        subfig2.plot(np.full(n_outliers, 2.26),
                    emp_mahal[-n_outliers:], '+k', markeredgewidth=1)
        subfig2.axes.set_xticklabels(('inliers', 'outliers'), size=15)
        subfig2.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
        subfig2.set_title("1. from non-robust estimates\n(Maximum Likelihood)")
        plt.yticks(())

        robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
        subfig3 = plt.subplot(2, 2, 4)
        subfig3.boxplot([robust_mahal[:-n_outliers], robust_mahal[-n_outliers:]],
                        widths=.25)
        subfig3.plot(np.full(n_samples - n_outliers, 1.26),
                    robust_mahal[:-n_outliers], '+k', markeredgewidth=1)
        subfig3.plot(np.full(n_outliers, 2.26),
                    robust_mahal[-n_outliers:], '+k', markeredgewidth=1)
        subfig3.axes.set_xticklabels(('inliers', 'outliers'), size=15)
        subfig3.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
        subfig3.set_title("2. from robust estimates\n(Minimum Covariance Determinant)")
        plt.yticks(())

        plt.show()
    """)
    selenium.run(cmd)

