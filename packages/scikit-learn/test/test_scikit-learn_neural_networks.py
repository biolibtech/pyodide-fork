from textwrap import dedent
import pytest

def test_sklearn_neural_networks_varying_regularization_in_multi_layer_perceptron(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import make_moons, make_circles, make_classification
        from sklearn.neural_network import MLPClassifier

        h = .02  # step size in the mesh

        alphas = np.logspace(-5, 3, 5)
        names = []
        for i in alphas:
            names.append('alpha ' + str(i))

        classifiers = []
        for i in alphas:
            classifiers.append(MLPClassifier(alpha=i, random_state=1))

        X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                random_state=0, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        datasets = [make_moons(noise=0.3, random_state=0),
                    make_circles(noise=0.2, factor=0.5, random_state=1),
                    linearly_separable]

        figure = plt.figure(figsize=(17, 9))
        i = 1
        # iterate over datasets
        for X, y in datasets:
            # preprocess dataset, split into training and test part
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

                # Plot also the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                        edgecolors='black', s=25)
                # and testing points
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                        alpha=0.6, edgecolors='black', s=25)

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                ax.set_title(name)
                ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                        size=15, horizontalalignment='right')
                i += 1

        figure.subplots_adjust(left=.02, right=.98)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_neural_networks_stochastic_learning_strategies_for_MLP(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import MinMaxScaler
        from sklearn import datasets

        # different learning rate schedules and momentum parameters
        params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
                'learning_rate_init': 0.2},
                {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
                'nesterovs_momentum': False, 'learning_rate_init': 0.2},
                {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
                'nesterovs_momentum': True, 'learning_rate_init': 0.2},
                {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
                'learning_rate_init': 0.2},
                {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
                'nesterovs_momentum': True, 'learning_rate_init': 0.2},
                {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
                'nesterovs_momentum': False, 'learning_rate_init': 0.2},
                {'solver': 'adam', 'learning_rate_init': 0.01}]

        labels = ["constant learning-rate", "constant with momentum",
                "constant with Nesterov's momentum",
                "inv-scaling learning-rate", "inv-scaling with momentum",
                "inv-scaling with Nesterov's momentum", "adam"]

        plot_args = [{'c': 'red', 'linestyle': '-'},
                    {'c': 'green', 'linestyle': '-'},
                    {'c': 'blue', 'linestyle': '-'},
                    {'c': 'red', 'linestyle': '--'},
                    {'c': 'green', 'linestyle': '--'},
                    {'c': 'blue', 'linestyle': '--'},
                    {'c': 'black', 'linestyle': '-'}]


        def plot_on_dataset(X, y, ax, name):
            # for each dataset, plot learning for each learning strategy
            print("\nlearning on dataset %s" % name)
            ax.set_title(name)
            X = MinMaxScaler().fit_transform(X)
            mlps = []
            if name == "digits":
                # digits is larger but converges fairly quickly
                max_iter = 15
            else:
                max_iter = 400

            for label, param in zip(labels, params):
                print("training: %s" % label)
                mlp = MLPClassifier(verbose=0, random_state=0,
                                    max_iter=max_iter, **param)
                mlp.fit(X, y)
                mlps.append(mlp)
                print("Training set score: %f" % mlp.score(X, y))
                print("Training set loss: %f" % mlp.loss_)
            for mlp, label, args in zip(mlps, labels, plot_args):
                    ax.plot(mlp.loss_curve_, label=label, **args)


        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # load / generate some toy datasets
        iris = datasets.load_iris()
        digits = datasets.load_digits()
        data_sets = [(iris.data, iris.target),
                    (digits.data, digits.target),
                    datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
                    datasets.make_moons(noise=0.3, random_state=0)]

        for ax, data, name in zip(axes.ravel(), data_sets, ['iris', 'digits',
                                                            'circles', 'moons']):
            plot_on_dataset(*data, ax=ax, name=name)

        fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
        plt.show()
    """)
    selenium.run(cmd)