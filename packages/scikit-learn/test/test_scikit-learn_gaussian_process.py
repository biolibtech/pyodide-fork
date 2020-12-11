from textwrap import dedent
import pytest


def test_sklearn_gaussian_process_classifier(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, DotProduct


        xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                            np.linspace(-3, 3, 50))
        rng = np.random.RandomState(0)
        X = rng.randn(200, 2)
        Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

        # fit the model
        plt.figure(figsize=(10, 5))
        kernels = [1.0 * RBF(length_scale=1.0), 1.0 * DotProduct(sigma_0=1.0)**2]
        for i, kernel in enumerate(kernels):
            clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X, Y)

            # plot the decision function for each datapoint on the grid
            Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
            Z = Z.reshape(xx.shape)

            plt.subplot(1, 2, i + 1)
            image = plt.imshow(Z, interpolation='nearest',
                            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                            aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
            contours = plt.contour(xx, yy, Z, levels=[0.5], linewidths=2,
                                colors=['k'])
            plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                        edgecolors=(0, 0, 0))
            plt.xticks(())
            plt.yticks(())
            plt.axis([-3, 3, -3, 3])
            plt.colorbar(image)
            plt.title("%s\n Log-Marginal-Likelihood:%.3f"
                    % (clf.kernel_, clf.log_marginal_likelihood(clf.kernel_.theta)),
                    fontsize=12)

        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_gaussian_process_kernel_ridge_vs_gaussian_process_classifier(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import time
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

        rng = np.random.RandomState(0)

        # Generate sample data
        X = 15 * rng.rand(100, 1)
        y = np.sin(X).ravel()
        y += 3 * (0.5 - rng.rand(X.shape[0]))  # add noise

        # Fit KernelRidge with parameter selection based on 5-fold cross validation
        param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                    "kernel": [ExpSineSquared(l, p)
                                for l in np.logspace(-2, 2, 10)
                                for p in np.logspace(0, 2, 10)]}
        kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
        stime = time.time()
        kr.fit(X, y)
        print("Time for KRR fitting: %.3f" % (time.time() - stime))

        gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
            + WhiteKernel(1e-1)
        gpr = GaussianProcessRegressor(kernel=gp_kernel)
        stime = time.time()
        gpr.fit(X, y)
        print("Time for GPR fitting: %.3f" % (time.time() - stime))

        # Predict using kernel ridge
        X_plot = np.linspace(0, 20, 10000)[:, None]
        stime = time.time()
        y_kr = kr.predict(X_plot)
        print("Time for KRR prediction: %.3f" % (time.time() - stime))

        # Predict using gaussian process regressor
        stime = time.time()
        y_gpr = gpr.predict(X_plot, return_std=False)
        print("Time for GPR prediction: %.3f" % (time.time() - stime))

        stime = time.time()
        y_gpr, y_std = gpr.predict(X_plot, return_std=True)
        print("Time for GPR prediction with standard-deviation: %.3f"
            % (time.time() - stime))

        # Plot results
        plt.figure(figsize=(10, 5))
        lw = 2
        plt.scatter(X, y, c='k', label='data')
        plt.plot(X_plot, np.sin(X_plot), color='navy', lw=lw, label='True')
        plt.plot(X_plot, y_kr, color='turquoise', lw=lw,
                label='KRR (%s)' % kr.best_params_)
        plt.plot(X_plot, y_gpr, color='darkorange', lw=lw,
                label='GPR (%s)' % gpr.kernel_)
        plt.fill_between(X_plot[:, 0], y_gpr - y_std, y_gpr + y_std, color='darkorange',
                        alpha=0.2)
        plt.xlabel('data')
        plt.ylabel('target')
        plt.xlim(0, 20)
        plt.ylim(-4, 4)
        plt.title('GPR versus Kernel Ridge')
        plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_gaussian_process_probability_prediction_gcp(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np

        from matplotlib import pyplot as plt

        from sklearn.metrics.classification import accuracy_score, log_loss
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF


        # Generate data
        train_size = 50
        rng = np.random.RandomState(0)
        X = rng.uniform(0, 5, 100)[:, np.newaxis]
        y = np.array(X[:, 0] > 2.5, dtype=int)

        # Specify Gaussian Processes with fixed and optimized hyperparameters
        gp_fix = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                        optimizer=None)
        gp_fix.fit(X[:train_size], y[:train_size])

        gp_opt = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        gp_opt.fit(X[:train_size], y[:train_size])

        print("Log Marginal Likelihood (initial): %.3f"
            % gp_fix.log_marginal_likelihood(gp_fix.kernel_.theta))
        print("Log Marginal Likelihood (optimized): %.3f"
            % gp_opt.log_marginal_likelihood(gp_opt.kernel_.theta))

        print("Accuracy: %.3f (initial) %.3f (optimized)"
            % (accuracy_score(y[:train_size], gp_fix.predict(X[:train_size])),
                accuracy_score(y[:train_size], gp_opt.predict(X[:train_size]))))
        print("Log-loss: %.3f (initial) %.3f (optimized)"
            % (log_loss(y[:train_size], gp_fix.predict_proba(X[:train_size])[:, 1]),
                log_loss(y[:train_size], gp_opt.predict_proba(X[:train_size])[:, 1])))


        # Plot posteriors
        plt.figure()
        plt.scatter(X[:train_size, 0], y[:train_size], c='k', label="Train data",
                    edgecolors=(0, 0, 0))
        plt.scatter(X[train_size:, 0], y[train_size:], c='g', label="Test data",
                    edgecolors=(0, 0, 0))
        X_ = np.linspace(0, 5, 100)
        plt.plot(X_, gp_fix.predict_proba(X_[:, np.newaxis])[:, 1], 'r',
                label="Initial kernel: %s" % gp_fix.kernel_)
        plt.plot(X_, gp_opt.predict_proba(X_[:, np.newaxis])[:, 1], 'b',
                label="Optimized kernel: %s" % gp_opt.kernel_)
        plt.xlabel("Feature")
        plt.ylabel("Class 1 probability")
        plt.xlim(0, 5)
        plt.ylim(-0.25, 1.5)
        plt.legend(loc="best")

        # Plot LML landscape
        plt.figure()
        theta0 = np.logspace(0, 8, 30)
        theta1 = np.logspace(-1, 1, 29)
        Theta0, Theta1 = np.meshgrid(theta0, theta1)
        LML = [[gp_opt.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))
                for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
        LML = np.array(LML).T
        plt.plot(np.exp(gp_fix.kernel_.theta)[0], np.exp(gp_fix.kernel_.theta)[1],
                'ko', zorder=10)
        plt.plot(np.exp(gp_opt.kernel_.theta)[0], np.exp(gp_opt.kernel_.theta)[1],
                'ko', zorder=10)
        plt.pcolor(Theta0, Theta1, LML)
        plt.xscale("log")
        plt.yscale("log")
        plt.colorbar()
        plt.xlabel("Magnitude")
        plt.ylabel("Length-scale")
        plt.title("Log-marginal-likelihood")

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_gaussian_process_regression(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from matplotlib import pyplot as plt

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

        np.random.seed(1)


        def f(x):
            return x * np.sin(x)

        # ----------------------------------------------------------------------
        #  First the noiseless case
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

        # Observations
        y = f(X).ravel()

        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        x = np.atleast_2d(np.linspace(0, 10, 1000)).T

        # Instantiate a Gaussian Process model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(X, y)

        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, sigma = gp.predict(x, return_std=True)

        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.figure()
        plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
        plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
        plt.plot(x, y_pred, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, 20)
        plt.legend(loc='upper left')

        # ----------------------------------------------------------------------
        # now the noisy case
        X = np.linspace(0.1, 9.9, 20)
        X = np.atleast_2d(X).T

        # Observations and noise
        y = f(X).ravel()
        dy = 0.5 + 1.0 * np.random.random(y.shape)
        noise = np.random.normal(0, dy)
        y += noise

        # Instantiate a Gaussian Process model
        gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                    n_restarts_optimizer=10)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(X, y)

        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, sigma = gp.predict(x, return_std=True)

        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.figure()
        plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
        plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
        plt.plot(x, y_pred, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, 20)
        plt.legend(loc='upper left')

        plt.show()
    """)
    selenium.run(cmd)
