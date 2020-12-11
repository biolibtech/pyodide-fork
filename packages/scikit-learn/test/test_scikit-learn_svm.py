def test_sklearn_support_vector_machines_non_linear_SVM(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import svm

        xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                            np.linspace(-3, 3, 500))
        np.random.seed(0)
        X = np.random.randn(300, 2)
        Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

        # fit the model
        clf = svm.NuSVC()
        clf.fit(X, Y)

        # plot the decision function for each datapoint on the grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.imshow(Z, interpolation='nearest',
                extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                origin='lower', cmap=plt.cm.PuOr_r)
        contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                            linetypes='--')
        plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                    edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis([-3, 3, -3, 3])
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_support_vector_machines_maximum_margin(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import svm
        from sklearn.datasets import make_blobs


        # we create 40 separable points
        X, y = make_blobs(n_samples=40, centers=2, random_state=6)

        # fit the model, don't regularize for illustration purposes
        clf = svm.SVC(kernel='linear', C=1000)
        clf.fit(X, y)

        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
        # plot support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_support_vector_machines_kernels(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import svm


        # Our dataset and targets
        X = np.c_[(.4, -.7),
                (-1.5, -1),
                (-1.4, -.9),
                (-1.3, -1.2),
                (-1.1, -.2),
                (-1.2, -.4),
                (-.5, 1.2),
                (-1.5, 2.1),
                (1, 1),
                # --
                (1.3, .8),
                (1.2, .5),
                (.2, -2),
                (.5, -2.4),
                (.2, -2.3),
                (0, -2.7),
                (1.3, 2.1)].T
        Y = [0] * 8 + [1] * 8

        # figure number
        fignum = 1

        # fit the model
        for kernel in ('linear', 'poly', 'rbf'):
            clf = svm.SVC(kernel=kernel, gamma=2)
            clf.fit(X, Y)

            # plot the line, the points, and the nearest vectors to the plane
            plt.figure(fignum, figsize=(4, 3))
            plt.clf()

            plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                        facecolors='none', zorder=10, edgecolors='k')
            plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                        edgecolors='k')

            plt.axis('tight')
            x_min = -3
            x_max = 3
            y_min = -3
            y_max = 3

            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(XX.shape)
            plt.figure(fignum, figsize=(4, 3))
            plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
            plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                        levels=[-.5, 0, .5])

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            plt.xticks(())
            plt.yticks(())
            fignum = fignum + 1
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_support_vector_machines_ANOVA(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectPercentile, chi2
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC


        # #############################################################################
        # Import some data to play with
        X, y = load_iris(return_X_y=True)
        # Add non-informative features
        np.random.seed(0)
        X = np.hstack((X, 2 * np.random.random((X.shape[0], 36))))

        # #############################################################################
        # Create a feature-selection transform, a scaler and an instance of SVM that we
        # combine together to have an full-blown estimator
        clf = Pipeline([('anova', SelectPercentile(chi2)),
                        ('scaler', StandardScaler()),
                        ('svc', SVC(gamma="auto"))])

        # #############################################################################
        # Plot the cross-validation score as a function of percentile of features
        score_means = list()
        score_stds = list()
        percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

        for percentile in percentiles:
            clf.set_params(anova__percentile=percentile)
            this_scores = cross_val_score(clf, X, y, cv=5)
            score_means.append(this_scores.mean())
            score_stds.append(this_scores.std())

        plt.errorbar(percentiles, score_means, np.array(score_stds))
        plt.title(
            'Performance of the SVM-Anova varying the percentile of features selected')
        plt.xticks(np.linspace(0, 100, 11, endpoint=True))
        plt.xlabel('Percentile')
        plt.ylabel('Accuracy Score')
        plt.axis('tight')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_support_vector_machines_with_RBF(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.font_manager
        from sklearn import svm

        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
        # Generate train data
        X = 0.3 * np.random.randn(100, 2)
        X_train = np.r_[X + 2, X - 2]
        # Generate some regular novel observations
        X = 0.3 * np.random.randn(20, 2)
        X_test = np.r_[X + 2, X - 2]
        # Generate some abnormal novel observations
        X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

        # fit the model
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

        # plot the line, the points, and the nearest vectors to the plane
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title("Novelty Detection")
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
            "error train: %d/200 ; errors novel regular: %d/40 ; "
            "errors novel abnormal: %d/40"
            % (n_error_train, n_error_test, n_error_outliers))
        plt.show()
    """)
    selenium.run(cmd)
