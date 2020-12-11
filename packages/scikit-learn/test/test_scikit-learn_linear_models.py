from textwrap import dedent
import pytest

def test_sklearn_linear_models_lasso_path_LARS(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import linear_model
        from sklearn import datasets

        diabetes = datasets.load_diabetes()
        X = diabetes.data
        y = diabetes.target

        print("Computing regularization path using the LARS ...")
        _, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

        xx = np.sum(np.abs(coefs.T), axis=1)
        xx /= xx[-1]

        plt.plot(xx, coefs.T)
        ymin, ymax = plt.ylim()
        plt.vlines(xx, ymin, ymax, linestyle='dashed')
        plt.xlabel('|coef| / max|coef|')
        plt.ylabel('Coefficients')
        plt.title('LASSO Path')
        plt.axis('tight')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_ridge_regularization(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import linear_model

        # X is the 10x10 Hilbert matrix
        X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
        y = np.ones(10)

        # #############################################################################
        # Compute paths

        n_alphas = 200
        alphas = np.logspace(-10, -2, n_alphas)

        coefs = []
        for a in alphas:
            ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
            ridge.fit(X, y)
            coefs.append(ridge.coef_)

        # #############################################################################
        # Display results

        ax = plt.gca()

        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Ridge coefficients as a function of the regularization')
        plt.axis('tight')
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_sgd_convex_loss_functions(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt


        def modified_huber_loss(y_true, y_pred):
            z = y_pred * y_true
            loss = -4 * z
            loss[z >= -1] = (1 - z[z >= -1]) ** 2
            loss[z >= 1.] = 0
            return loss


        xmin, xmax = -4, 4
        xx = np.linspace(xmin, xmax, 100)
        lw = 2
        plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], color='gold', lw=lw,
                label="Zero-one loss")
        plt.plot(xx, np.where(xx < 1, 1 - xx, 0), color='teal', lw=lw,
                label="Hinge loss")
        plt.plot(xx, -np.minimum(xx, 0), color='yellowgreen', lw=lw,
                label="Perceptron loss")
        plt.plot(xx, np.log2(1 + np.exp(-xx)), color='cornflowerblue', lw=lw,
                label="Log loss")
        plt.plot(xx, np.where(xx < 1, 1 - xx, 0) ** 2, color='orange', lw=lw,
                label="Squared hinge loss")
        plt.plot(xx, modified_huber_loss(xx, 1), color='darkorchid', lw=lw,
                linestyle='--', label="Modified Huber loss")
        plt.ylim((0, 8))
        plt.legend(loc="upper right")
        plt.xlabel(r"Decision function $f(x)$")
        plt.ylabel("$L(y=1, f(x))$")
        plt.show()

    """)
    selenium.run(cmd)

def test_sklearn_linear_models_ridge_l2(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np

        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        clf = Ridge()

        X, y, w = make_regression(n_samples=10, n_features=10, coef=True,
                                random_state=1, bias=3.5)

        coefs = []
        errors = []

        alphas = np.logspace(-6, 6, 200)

        # Train the model with different regularisation strengths
        for a in alphas:
            clf.set_params(alpha=a)
            clf.fit(X, y)
            coefs.append(clf.coef_)
            errors.append(mean_squared_error(clf.coef_, w))

        # Display results
        plt.figure(figsize=(20, 6))

        plt.subplot(121)
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Ridge coefficients as a function of the regularization')
        plt.axis('tight')

        plt.subplot(122)
        ax = plt.gca()
        ax.plot(alphas, errors)
        ax.set_xscale('log')
        plt.xlabel('alpha')
        plt.ylabel('error')
        plt.title('Coefficient error as a function of the regularization')
        plt.axis('tight')

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_l1_regularization(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from time import time
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import linear_model
        from sklearn import datasets
        from sklearn.svm import l1_min_c

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        X = X[y != 2]
        y = y[y != 2]

        X /= X.max()  # Normalize X to speed-up convergence

        # #############################################################################
        # Demo path functions

        cs = l1_min_c(X, y, loss='log') * np.logspace(0, 7, 16)


        print("Computing regularization path ...")
        start = time()
        clf = linear_model.LogisticRegression(penalty='l1', solver='saga',
                                            tol=1e-6, max_iter=int(1e6),
                                            warm_start=True)
        coefs_ = []
        for c in cs:
            clf.set_params(C=c)
            clf.fit(X, y)
            coefs_.append(clf.coef_.ravel().copy())
        print("This took %0.3fs" % (time() - start))

        coefs_ = np.array(coefs_)
        plt.plot(np.log10(cs), coefs_, marker='o')
        ymin, ymax = plt.ylim()
        plt.xlabel('log(C)')
        plt.ylabel('Coefficients')
        plt.title('Logistic Regression Path')
        plt.axis('tight')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_l1_regularization(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import linear_model

        # General a toy dataset:s it's just a straight line with some Gaussian noise:
        xmin, xmax = -5, 5
        n_samples = 100
        np.random.seed(0)
        X = np.random.normal(size=n_samples)
        y = (X > 0).astype(np.float)
        X[X > 0] *= 4
        X += .3 * np.random.normal(size=n_samples)

        X = X[:, np.newaxis]

        # Fit the classifier
        clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs')
        clf.fit(X, y)

        # and plot the result
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.scatter(X.ravel(), y, color='black', zorder=20)
        X_test = np.linspace(-5, 10, 300)


        def model(x):
            return 1 / (1 + np.exp(-x))


        loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
        plt.plot(X_test, loss, color='red', linewidth=3)

        ols = linear_model.LinearRegression()
        ols.fit(X, y)
        plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
        plt.axhline(.5, color='.5')

        plt.ylabel('y')
        plt.xlabel('X')
        plt.xticks(range(-5, 10))
        plt.yticks([0, 0.5, 1])
        plt.ylim(-.25, 1.25)
        plt.xlim(-4, 10)
        plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
                loc="lower right", fontsize='small')
        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_logistic_regression_3_class(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LogisticRegression
        from sklearn import datasets

        # import some data to play with
        iris = datasets.load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        Y = iris.target

        logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

        # Create an instance of Logistic Regression Classifier and fit the data.
        logreg.fit(X, Y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_sparsity(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D

        from sklearn import datasets, linear_model

        diabetes = datasets.load_diabetes()
        indices = (0, 1)

        X_train = diabetes.data[:-20, indices]
        X_test = diabetes.data[-20:, indices]
        y_train = diabetes.target[:-20]
        y_test = diabetes.target[-20:]

        ols = linear_model.LinearRegression()
        ols.fit(X_train, y_train)


        # #############################################################################
        # Plot the figure
        def plot_figs(fig_num, elev, azim, X_train, clf):
            fig = plt.figure(fig_num, figsize=(4, 3))
            plt.clf()
            ax = Axes3D(fig, elev=elev, azim=azim)

            ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')
            ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
                            np.array([[-.1, .15], [-.1, .15]]),
                            clf.predict(np.array([[-.1, -.1, .15, .15],
                                                [-.1, .15, -.1, .15]]).T
                                        ).reshape((2, 2)),
                            alpha=.5)
            ax.set_xlabel('X_1')
            ax.set_ylabel('X_2')
            ax.set_zlabel('Y')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])

        #Generate the three different figures from different views
        elev = 43.5
        azim = -110
        plot_figs(1, elev, azim, X_train, ols)

        elev = -.5
        azim = 0
        plot_figs(2, elev, azim, X_train, ols)

        elev = -.5
        azim = 90
        plot_figs(3, elev, azim, X_train, ols)

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_huber_vs_ridge(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.datasets import make_regression
        from sklearn.linear_model import HuberRegressor, Ridge

        # Generate toy data.
        rng = np.random.RandomState(0)
        X, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=4.0,
                            bias=100.0)

        # Add four strong outliers to the dataset.
        X_outliers = rng.normal(0, 0.5, size=(4, 1))
        y_outliers = rng.normal(0, 2.0, size=4)
        X_outliers[:2, :] += X.max() + X.mean() / 4.
        X_outliers[2:, :] += X.min() - X.mean() / 4.
        y_outliers[:2] += y.min() - y.mean() / 4.
        y_outliers[2:] += y.max() + y.mean() / 4.
        X = np.vstack((X, X_outliers))
        y = np.concatenate((y, y_outliers))
        plt.plot(X, y, 'b.')

        # Fit the huber regressor over a series of epsilon values.
        colors = ['r-', 'b-', 'y-', 'm-']

        x = np.linspace(X.min(), X.max(), 7)
        epsilon_values = [1.35, 1.5, 1.75, 1.9]
        for k, epsilon in enumerate(epsilon_values):
            huber = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,
                                epsilon=epsilon)
            huber.fit(X, y)
            coef_ = huber.coef_ * x + huber.intercept_
            plt.plot(x, coef_, colors[k], label="huber loss, %s" % epsilon)

        # Fit a ridge regressor to compare it to huber regressor.
        ridge = Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True)
        ridge.fit(X, y)
        coef_ridge = ridge.coef_
        coef_ = ridge.coef_ * x + ridge.intercept_
        plt.plot(x, coef_, 'g-', label="ridge regression")

        plt.title("Comparison of HuberRegressor vs Ridge")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend(loc=0)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_lasso_and_elastic_net(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from itertools import cycle
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.linear_model import lasso_path, enet_path
        from sklearn import datasets

        diabetes = datasets.load_diabetes()
        X = diabetes.data
        y = diabetes.target

        X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

        # Compute paths

        eps = 5e-3  # the smaller it is the longer is the path

        print("Computing regularization path using the lasso...")
        alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

        print("Computing regularization path using the positive lasso...")
        alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
            X, y, eps, positive=True, fit_intercept=False)
        print("Computing regularization path using the elastic net...")
        alphas_enet, coefs_enet, _ = enet_path(
            X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

        print("Computing regularization path using the positive elastic net...")
        alphas_positive_enet, coefs_positive_enet, _ = enet_path(
            X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

        # Display results

        plt.figure(1)
        colors = cycle(['b', 'r', 'g', 'c', 'k'])
        neg_log_alphas_lasso = -np.log10(alphas_lasso)
        neg_log_alphas_enet = -np.log10(alphas_enet)
        for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
            l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
            l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('Lasso and Elastic-Net Paths')
        plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
        plt.axis('tight')


        plt.figure(2)
        neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
        for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
            l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
            l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('Lasso and positive Lasso')
        plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
        plt.axis('tight')


        plt.figure(3)
        neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
        for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
            l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
            l2 = plt.plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)

        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('Elastic-Net and positive Elastic-Net')
        plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
                loc='lower left')
        plt.axis('tight')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_robust_linear_estimator_fitting(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from matplotlib import pyplot as plt
        import numpy as np

        from sklearn.linear_model import (
            LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        np.random.seed(42)

        X = np.random.normal(size=400)
        y = np.sin(X)
        # Make sure that it X is 2D
        X = X[:, np.newaxis]

        X_test = np.random.normal(size=200)
        y_test = np.sin(X_test)
        X_test = X_test[:, np.newaxis]

        y_errors = y.copy()
        y_errors[::3] = 3

        X_errors = X.copy()
        X_errors[::3] = 3

        y_errors_large = y.copy()
        y_errors_large[::3] = 10

        X_errors_large = X.copy()
        X_errors_large[::3] = 10

        estimators = [('OLS', LinearRegression()),
                    ('Theil-Sen', TheilSenRegressor(random_state=42)),
                    ('RANSAC', RANSACRegressor(random_state=42)),
                    ('HuberRegressor', HuberRegressor())]
        colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black'}
        linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--'}
        lw = 3

        x_plot = np.linspace(X.min(), X.max())
        for title, this_X, this_y in [
                ('Modeling Errors Only', X, y),
                ('Corrupt X, Small Deviants', X_errors, y),
                ('Corrupt y, Small Deviants', X, y_errors),
                ('Corrupt X, Large Deviants', X_errors_large, y),
                ('Corrupt y, Large Deviants', X, y_errors_large)]:
            plt.figure(figsize=(5, 4))
            plt.plot(this_X[:, 0], this_y, 'b+')

            for name, estimator in estimators:
                model = make_pipeline(PolynomialFeatures(3), estimator)
                model.fit(this_X, this_y)
                mse = mean_squared_error(model.predict(X_test), y_test)
                y_plot = model.predict(x_plot[:, np.newaxis])
                plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                        linewidth=lw, label='%s: error = %.3f' % (name, mse))

            legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
            legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                                prop=dict(size='x-small'))
            plt.xlim(-4, 10.2)
            plt.ylim(-2, 10.2)
            plt.title(title)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_linear_models_bayesian_ridge_regression(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats

        from sklearn.linear_model import BayesianRidge, LinearRegression

        # #############################################################################
        # Generating simulated data with Gaussian weights
        np.random.seed(0)
        n_samples, n_features = 100, 100
        X = np.random.randn(n_samples, n_features)  # Create Gaussian data
        # Create weights with a precision lambda_ of 4.
        lambda_ = 4.
        w = np.zeros(n_features)
        # Only keep 10 weights of interest
        relevant_features = np.random.randint(0, n_features, 10)
        for i in relevant_features:
            w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
        # Create noise with a precision alpha of 50.
        alpha_ = 50.
        noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
        # Create the target
        y = np.dot(X, w) + noise

        # #############################################################################
        # Fit the Bayesian Ridge Regression and an OLS for comparison
        clf = BayesianRidge(compute_score=True)
        clf.fit(X, y)

        ols = LinearRegression()
        ols.fit(X, y)

        # #############################################################################
        # Plot true weights, estimated weights, histogram of the weights, and
        # predictions with standard deviations
        lw = 2
        plt.figure(figsize=(6, 5))
        plt.title("Weights of the model")
        plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
                label="Bayesian Ridge estimate")
        plt.plot(w, color='gold', linewidth=lw, label="Ground truth")
        plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")
        plt.xlabel("Features")
        plt.ylabel("Values of the weights")
        plt.legend(loc="best", prop=dict(size=12))

        plt.figure(figsize=(6, 5))
        plt.title("Histogram of the weights")
        plt.hist(clf.coef_, bins=n_features, color='gold', log=True,
                edgecolor='black')
        plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
                    color='navy', label="Relevant features")
        plt.ylabel("Features")
        plt.xlabel("Values of the weights")
        plt.legend(loc="upper left")

        plt.figure(figsize=(6, 5))
        plt.title("Marginal log-likelihood")
        plt.plot(clf.scores_, color='navy', linewidth=lw)
        plt.ylabel("Score")
        plt.xlabel("Iterations")


        # Plotting some predictions for polynomial regression
        def f(x, noise_amount):
            y = np.sqrt(x) * np.sin(x)
            noise = np.random.normal(0, 1, len(x))
            return y + noise_amount * noise


        degree = 10
        X = np.linspace(0, 10, 100)
        y = f(X, noise_amount=0.1)
        clf_poly = BayesianRidge()
        clf_poly.fit(np.vander(X, degree), y)

        X_plot = np.linspace(0, 11, 25)
        y_plot = f(X_plot, noise_amount=0)
        y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
        plt.figure(figsize=(6, 5))
        plt.errorbar(X_plot, y_mean, y_std, color='navy',
                    label="Polynomial Bayesian Ridge Regression", linewidth=lw)
        plt.plot(X_plot, y_plot, color='gold', linewidth=lw,
                label="Ground Truth")
        plt.ylabel("Output y")
        plt.xlabel("Feature X")
        plt.legend(loc="lower left")
        plt.show()
    """)
    selenium.run(cmd)

