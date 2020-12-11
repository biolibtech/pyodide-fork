from textwrap import dedent
import pytest

def test_sklearn_ensemble_decision_tree_regression_adaboost(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import AdaBoostRegressor

        # Create the dataset
        rng = np.random.RandomState(1)
        X = np.linspace(0, 6, 100)[:, np.newaxis]
        y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

        # Fit regression model
        regr_1 = DecisionTreeRegressor(max_depth=4)

        regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                n_estimators=300, random_state=rng)

        regr_1.fit(X, y)
        regr_2.fit(X, y)

        # Predict
        y_1 = regr_1.predict(X)
        y_2 = regr_2.predict(X)

        # Plot the results
        plt.figure()
        plt.scatter(X, y, c="k", label="training samples")
        plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
        plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Boosted Decision Tree Regression")
        plt.legend()
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_ensemble_parallel_forest(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from time import time
        import matplotlib.pyplot as plt

        from sklearn.datasets import fetch_olivetti_faces
        from sklearn.ensemble import ExtraTreesClassifier

        # Number of cores to use to perform parallel fitting of the forest model
        n_jobs = 1

        # Load the faces dataset
        data = fetch_olivetti_faces()
        X = data.images.reshape((len(data.images), -1))
        y = data.target

        mask = y < 5  # Limit to 5 classes
        X = X[mask]
        y = y[mask]

        # Build a forest and compute the pixel importances
        print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
        t0 = time()
        forest = ExtraTreesClassifier(n_estimators=1000,
                                    max_features=128,
                                    n_jobs=n_jobs,
                                    random_state=0)

        forest.fit(X, y)
        print("done in %0.3fs" % (time() - t0))
        importances = forest.feature_importances_
        importances = importances.reshape(data.images[0].shape)

        # Plot pixel importances
        plt.matshow(importances, cmap=plt.cm.hot)
        plt.title("Pixel importances with forests of trees")
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_ensemble_feature_importance(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.datasets import make_classification
        from sklearn.ensemble import ExtraTreesClassifier

        # Build a classification task using 3 informative features
        X, y = make_classification(n_samples=1000,
                                n_features=10,
                                n_informative=3,
                                n_redundant=0,
                                n_repeated=0,
                                n_classes=2,
                                random_state=0,
                                shuffle=False)

        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250,
                                    random_state=0)

        forest.fit(X, y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_ensemble_decision_boundaries(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from itertools import product

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import datasets
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import VotingClassifier

        # Loading some example data
        iris = datasets.load_iris()
        X = iris.data[:, [0, 2]]
        y = iris.target

        # Training classifiers
        clf1 = DecisionTreeClassifier(max_depth=4)
        clf2 = KNeighborsClassifier(n_neighbors=7)
        clf3 = SVC(gamma=.1, kernel='rbf', probability=True)
        eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                            ('svc', clf3)],
                                voting='soft', weights=[2, 1, 2])

        clf1.fit(X, y)
        clf2.fit(X, y)
        clf3.fit(X, y)
        eclf.fit(X, y)

        # Plotting decision regions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

        for idx, clf, tt in zip(product([0, 1], [0, 1]),
                                [clf1, clf2, clf3, eclf],
                                ['Decision Tree (depth=4)', 'KNN (k=7)',
                                'Kernel SVM', 'Soft Voting']):

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
            axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                        s=20, edgecolor='k')
            axarr[idx[0], idx[1]].set_title(tt)

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_ensemble_gradient_boosting(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.ensemble import GradientBoostingRegressor

        np.random.seed(1)


        def f(x):
            return x * np.sin(x)

        #----------------------------------------------------------------------
        #  First the noiseless case
        X = np.atleast_2d(np.random.uniform(0, 10.0, size=100)).T
        X = X.astype(np.float32)

        # Observations
        y = f(X).ravel()

        dy = 1.5 + 1.0 * np.random.random(y.shape)
        noise = np.random.normal(0, dy)
        y += noise
        y = y.astype(np.float32)

        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
        xx = xx.astype(np.float32)

        alpha = 0.95

        clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                        n_estimators=250, max_depth=3,
                                        learning_rate=.1, min_samples_leaf=9,
                                        min_samples_split=9)

        clf.fit(X, y)

        # Make the prediction on the meshed x-axis
        y_upper = clf.predict(xx)

        clf.set_params(alpha=1.0 - alpha)
        clf.fit(X, y)

        # Make the prediction on the meshed x-axis
        y_lower = clf.predict(xx)

        clf.set_params(loss='ls')
        clf.fit(X, y)

        # Make the prediction on the meshed x-axis
        y_pred = clf.predict(xx)

        # Plot the function, the prediction and the 90% confidence interval based on
        # the MSE
        fig = plt.figure()
        plt.plot(xx, f(xx), 'g:', label=r'$f(x) = x\,\sin(x)$')
        plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
        plt.plot(xx, y_pred, 'r-', label=u'Prediction')
        plt.plot(xx, y_upper, 'k-')
        plt.plot(xx, y_lower, 'k-')
        plt.fill(np.concatenate([xx, xx[::-1]]),
                np.concatenate([y_upper, y_lower[::-1]]),
                alpha=.5, fc='b', ec='None', label='90% prediction interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, 20)
        plt.legend(loc='upper left')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_ensemble_gradient_boosting_regularization(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import ensemble
        from sklearn import datasets


        X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
        X = X.astype(np.float32)

        # map labels from {-1, 1} to {0, 1}
        labels, y = np.unique(y, return_inverse=True)

        X_train, X_test = X[:2000], X[2000:]
        y_train, y_test = y[:2000], y[2000:]

        original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                        'min_samples_split': 5}

        plt.figure()

        for label, color, setting in [('No shrinkage', 'orange',
                                    {'learning_rate': 1.0, 'subsample': 1.0}),
                                    ('learning_rate=0.1', 'turquoise',
                                    {'learning_rate': 0.1, 'subsample': 1.0}),
                                    ('subsample=0.5', 'blue',
                                    {'learning_rate': 1.0, 'subsample': 0.5}),
                                    ('learning_rate=0.1, subsample=0.5', 'gray',
                                    {'learning_rate': 0.1, 'subsample': 0.5}),
                                    ('learning_rate=0.1, max_features=2', 'magenta',
                                    {'learning_rate': 0.1, 'max_features': 2})]:
            params = dict(original_params)
            params.update(setting)

            clf = ensemble.GradientBoostingClassifier(**params)
            clf.fit(X_train, y_train)

            # compute test set deviance
            test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

            for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
                # clf.loss_ assumes that y_test[i] in {0, 1}
                test_deviance[i] = clf.loss_(y_test, y_pred)

            plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
                    '-', color=color, label=label)

        plt.legend(loc='upper left')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Test Set Deviance')

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_ensemble_gradient_boosting_out_of_bag(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import ensemble
        from sklearn.model_selection import KFold
        from sklearn.model_selection import train_test_split


        # Generate data (adapted from G. Ridgeway's gbm example)
        n_samples = 1000
        random_state = np.random.RandomState(13)
        x1 = random_state.uniform(size=n_samples)
        x2 = random_state.uniform(size=n_samples)
        x3 = random_state.randint(0, 4, size=n_samples)

        p = 1 / (1.0 + np.exp(-(np.sin(3 * x1) - 4 * x2 + x3)))
        y = random_state.binomial(1, p, size=n_samples)

        X = np.c_[x1, x2, x3]

        X = X.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                            random_state=9)

        # Fit classifier with out-of-bag estimates
        params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
                'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
        clf = ensemble.GradientBoostingClassifier(**params)

        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print("Accuracy: {:.4f}".format(acc))

        n_estimators = params['n_estimators']
        x = np.arange(n_estimators) + 1


        def heldout_score(clf, X_test, y_test):
            score = np.zeros((n_estimators,), dtype=np.float64)
            for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
                score[i] = clf.loss_(y_test, y_pred)
            return score


        def cv_estimate(n_splits=None):
            cv = KFold(n_splits=n_splits)
            cv_clf = ensemble.GradientBoostingClassifier(**params)
            val_scores = np.zeros((n_estimators,), dtype=np.float64)
            for train, test in cv.split(X_train, y_train):
                cv_clf.fit(X_train[train], y_train[train])
                val_scores += heldout_score(cv_clf, X_train[test], y_train[test])
            val_scores /= n_splits
            return val_scores


        # Estimate best n_estimator using cross-validation
        cv_score = cv_estimate(3)

        # Compute best n_estimator for test data
        test_score = heldout_score(clf, X_test, y_test)

        # negative cumulative sum of oob improvements
        cumsum = -np.cumsum(clf.oob_improvement_)

        # min loss according to OOB
        oob_best_iter = x[np.argmin(cumsum)]

        # min loss according to test (normalize such that first loss is 0)
        test_score -= test_score[0]
        test_best_iter = x[np.argmin(test_score)]

        # min loss according to cv (normalize such that first loss is 0)
        cv_score -= cv_score[0]
        cv_best_iter = x[np.argmin(cv_score)]

        # color brew for the three curves
        oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))
        test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))
        cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))

        # plot curves and vertical lines for best iterations
        plt.plot(x, cumsum, label='OOB loss', color=oob_color)
        plt.plot(x, test_score, label='Test loss', color=test_color)
        plt.plot(x, cv_score, label='CV loss', color=cv_color)
        plt.axvline(x=oob_best_iter, color=oob_color)
        plt.axvline(x=test_best_iter, color=test_color)
        plt.axvline(x=cv_best_iter, color=cv_color)

        # add three vertical lines to xticks
        xticks = plt.xticks()
        xticks_pos = np.array(xticks[0].tolist() +
                            [oob_best_iter, cv_best_iter, test_best_iter])
        xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) +
                                ['OOB', 'CV', 'Test'])
        ind = np.argsort(xticks_pos)
        xticks_pos = xticks_pos[ind]
        xticks_label = xticks_label[ind]
        plt.xticks(xticks_pos, xticks_label)

        plt.legend(loc='upper right')
        plt.ylabel('normalized loss')
        plt.xlabel('number of iterations')

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_ensemble_gradient_boosting_early_stopping(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import time

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import ensemble
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        print(__doc__)

        data_list = [datasets.load_iris(), datasets.load_digits()]
        data_list = [(d.data, d.target) for d in data_list]
        data_list += [datasets.make_hastie_10_2()]
        names = ['Iris Data', 'Digits Data', 'Hastie Data']

        n_gb = []
        score_gb = []
        time_gb = []
        n_gbes = []
        score_gbes = []
        time_gbes = []

        n_estimators = 500

        for X, y in data_list:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                random_state=0)

            # We specify that if the scores don't improve by atleast 0.01 for the last
            # 10 stages, stop fitting additional stages
            gbes = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                                    validation_fraction=0.2,
                                                    n_iter_no_change=5, tol=0.01,
                                                    random_state=0)
            gb = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                                    random_state=0)
            start = time.time()
            gb.fit(X_train, y_train)
            time_gb.append(time.time() - start)

            start = time.time()
            gbes.fit(X_train, y_train)
            time_gbes.append(time.time() - start)

            score_gb.append(gb.score(X_test, y_test))
            score_gbes.append(gbes.score(X_test, y_test))

            n_gb.append(gb.n_estimators_)
            n_gbes.append(gbes.n_estimators_)

        bar_width = 0.2
        n = len(data_list)
        index = np.arange(0, n * bar_width, bar_width) * 2.5
        index = index[0:n]
    """)
    selenium.run(cmd)