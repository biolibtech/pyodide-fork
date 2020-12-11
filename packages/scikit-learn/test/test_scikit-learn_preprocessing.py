from textwrap import dedent
import pytest

def test_sklearn_preprocessing_function_transformer(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np

        from sklearn.model_selection import train_test_split
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer


        def _generate_vector(shift=0.5, noise=15):
            return np.arange(1000) + (np.random.rand(1000) - shift) * noise


        def generate_dataset():
            return np.vstack((
                np.vstack((
                    _generate_vector(),
                    _generate_vector() + 100,
                )).T,
                np.vstack((
                    _generate_vector(),
                    _generate_vector(),
                )).T,
            )), np.hstack((np.zeros(1000), np.ones(1000)))


        def all_but_first_column(X):
            return X[:, 1:]


        def drop_first_component(X, y):
            pipeline = make_pipeline(
                PCA(), FunctionTransformer(all_but_first_column),
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            pipeline.fit(X_train, y_train)
            return pipeline.transform(X_test), y_test


        if __name__ == '__main__':
            X, y = generate_dataset()
            lw = 0
            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=y, lw=lw)
            plt.figure()
            X_transformed, y_transformed = drop_first_component(*generate_dataset())
            plt.scatter(
                X_transformed[:, 0],
                np.zeros(len(X_transformed)),
                c=y_transformed,
                lw=lw,
                s=60
            )
            plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_preprocessing_discrete_bins(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import KBinsDiscretizer
        from sklearn.tree import DecisionTreeRegressor

        print(__doc__)

        # construct the dataset
        rnd = np.random.RandomState(42)
        X = rnd.uniform(-3, 3, size=100)
        y = np.sin(X) + rnd.normal(size=len(X)) / 3
        X = X.reshape(-1, 1)

        # transform the dataset with KBinsDiscretizer
        enc = KBinsDiscretizer(n_bins=10, encode='onehot')
        X_binned = enc.fit_transform(X)

        # predict with original dataset
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
        line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        ax1.plot(line, reg.predict(line), linewidth=2, color='green',
                label="linear regression")
        reg = DecisionTreeRegressor(min_samples_split=3, random_state=0).fit(X, y)
        ax1.plot(line, reg.predict(line), linewidth=2, color='red',
                label="decision tree")
        ax1.plot(X[:, 0], y, 'o', c='k')
        ax1.legend(loc="best")
        ax1.set_ylabel("Regression output")
        ax1.set_xlabel("Input feature")
        ax1.set_title("Result before discretization")

        # predict with transformed dataset
        line_binned = enc.transform(line)
        reg = LinearRegression().fit(X_binned, y)
        ax2.plot(line, reg.predict(line_binned), linewidth=2, color='green',
                linestyle='-', label='linear regression')
        reg = DecisionTreeRegressor(min_samples_split=3,
                                    random_state=0).fit(X_binned, y)
        ax2.plot(line, reg.predict(line_binned), linewidth=2, color='red',
                linestyle=':', label='decision tree')
        ax2.plot(X[:, 0], y, 'o', c='k')
        ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1, alpha=.2)
        ax2.legend(loc="best")
        ax2.set_xlabel("Input feature")
        ax2.set_title("Result after discretization")

        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_preprocessing_different_discrete_binning_strategies(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.preprocessing import KBinsDiscretizer
        from sklearn.datasets import make_blobs

        print(__doc__)

        strategies = ['uniform', 'quantile', 'kmeans']

        n_samples = 200
        centers_0 = np.array([[0, 0], [0, 5], [2, 4], [8, 8]])
        centers_1 = np.array([[0, 0], [3, 1]])

        # construct the datasets
        random_state = 42
        X_list = [
            np.random.RandomState(random_state).uniform(-3, 3, size=(n_samples, 2)),
            make_blobs(n_samples=[n_samples // 10, n_samples * 4 // 10,
                                n_samples // 10, n_samples * 4 // 10],
                    cluster_std=0.5, centers=centers_0,
                    random_state=random_state)[0],
            make_blobs(n_samples=[n_samples // 5, n_samples * 4 // 5],
                    cluster_std=0.5, centers=centers_1,
                    random_state=random_state)[0],
        ]

        figure = plt.figure(figsize=(14, 9))
        i = 1
        for ds_cnt, X in enumerate(X_list):

            ax = plt.subplot(len(X_list), len(strategies) + 1, i)
            ax.scatter(X[:, 0], X[:, 1], edgecolors='k')
            if ds_cnt == 0:
                ax.set_title("Input data", size=14)

            xx, yy = np.meshgrid(
                np.linspace(X[:, 0].min(), X[:, 0].max(), 300),
                np.linspace(X[:, 1].min(), X[:, 1].max(), 300))
            grid = np.c_[xx.ravel(), yy.ravel()]

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

            i += 1
            # transform the dataset with KBinsDiscretizer
            for strategy in strategies:
                enc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy=strategy)
                enc.fit(X)
                grid_encoded = enc.transform(grid)

                ax = plt.subplot(len(X_list), len(strategies) + 1, i)

                # horizontal stripes
                horizontal = grid_encoded[:, 0].reshape(xx.shape)
                ax.contourf(xx, yy, horizontal, alpha=.5)
                # vertical stripes
                vertical = grid_encoded[:, 1].reshape(xx.shape)
                ax.contourf(xx, yy, vertical, alpha=.5)

                ax.scatter(X[:, 0], X[:, 1], edgecolors='k')
                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                if ds_cnt == 0:
                    ax.set_title("strategy='%s'" % (strategy, ), size=14)

                i += 1

        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_preprocessing_map_data_to_normal_distribution(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.preprocessing import PowerTransformer
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.model_selection import train_test_split

        print(__doc__)


        N_SAMPLES = 1000
        FONT_SIZE = 6
        BINS = 30


        rng = np.random.RandomState(304)
        bc = PowerTransformer(method='box-cox')
        yj = PowerTransformer(method='yeo-johnson')
        qt = QuantileTransformer(output_distribution='normal', random_state=rng)
        size = (N_SAMPLES, 1)


        # lognormal distribution
        X_lognormal = rng.lognormal(size=size)

        # chi-squared distribution
        df = 3
        X_chisq = rng.chisquare(df=df, size=size)

        # weibull distribution
        a = 50
        X_weibull = rng.weibull(a=a, size=size)

        # gaussian distribution
        loc = 100
        X_gaussian = rng.normal(loc=loc, size=size)

        # uniform distribution
        X_uniform = rng.uniform(low=0, high=1, size=size)

        # bimodal distribution
        loc_a, loc_b = 100, 105
        X_a, X_b = rng.normal(loc=loc_a, size=size), rng.normal(loc=loc_b, size=size)
        X_bimodal = np.concatenate([X_a, X_b], axis=0)


        # create plots
        distributions = [
            ('Lognormal', X_lognormal),
            ('Chi-squared', X_chisq),
            ('Weibull', X_weibull),
            ('Gaussian', X_gaussian),
            ('Uniform', X_uniform),
            ('Bimodal', X_bimodal)
        ]

        colors = ['firebrick', 'darkorange', 'goldenrod',
                'seagreen', 'royalblue', 'darkorchid']

        fig, axes = plt.subplots(nrows=8, ncols=3, figsize=plt.figaspect(2))
        axes = axes.flatten()
        axes_idxs = [(0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8, 11), (12, 15, 18, 21),
                    (13, 16, 19, 22), (14, 17, 20, 23)]
        axes_list = [(axes[i], axes[j], axes[k], axes[l])
                    for (i, j, k, l) in axes_idxs]


        for distribution, color, axes in zip(distributions, colors, axes_list):
            name, X = distribution
            X_train, X_test = train_test_split(X, test_size=.5)

            # perform power transforms and quantile transform
            X_trans_bc = bc.fit(X_train).transform(X_test)
            lmbda_bc = round(bc.lambdas_[0], 2)
            X_trans_yj = yj.fit(X_train).transform(X_test)
            lmbda_yj = round(yj.lambdas_[0], 2)
            X_trans_qt = qt.fit(X_train).transform(X_test)

            ax_original, ax_bc, ax_yj, ax_qt = axes

            ax_original.hist(X_train, color=color, bins=BINS)
            ax_original.set_title(name, fontsize=FONT_SIZE)
            ax_original.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

            for ax, X_trans, meth_name, lmbda in zip(
                    (ax_bc, ax_yj, ax_qt),
                    (X_trans_bc, X_trans_yj, X_trans_qt),
                    ('Box-Cox', 'Yeo-Johnson', 'Quantile transform'),
                    (lmbda_bc, lmbda_yj, None)):
                ax.hist(X_trans, color=color, bins=BINS)
                title = 'After {}'.format(meth_name)
                if lmbda is not None:
                    title += r'\n$\lambda$ = {}'.format(lmbda)
                ax.set_title(title, fontsize=FONT_SIZE)
                ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
                ax.set_xlim([-3.5, 3.5])


        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_preprocessing_feature_scaling(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from __future__ import print_function
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.naive_bayes import GaussianNB
        from sklearn import metrics
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_wine
        from sklearn.pipeline import make_pipeline
        print(__doc__)

        # Code source: Tyler Lanigan <tylerlanigan@gmail.com>
        #              Sebastian Raschka <mail@sebastianraschka.com>

        # License: BSD 3 clause

        RANDOM_STATE = 42
        FIG_SIZE = (10, 7)


        features, target = load_wine(return_X_y=True)

        # Make a train/test split using 30% test size
        X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                            test_size=0.30,
                                                            random_state=RANDOM_STATE)

        # Fit to data and predict using pipelined GNB and PCA.
        unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
        unscaled_clf.fit(X_train, y_train)
        pred_test = unscaled_clf.predict(X_test)

        # Fit to data and predict using pipelined scaling, GNB and PCA.
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
        std_clf.fit(X_train, y_train)
        pred_test_std = std_clf.predict(X_test)

        # Show prediction accuracies in scaled and unscaled data.
        print('\nPrediction accuracy for the normal test dataset with PCA')
        print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

        print('\nPrediction accuracy for the standardized test dataset with PCA')
        print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

        # Extract PCA from pipeline
        pca = unscaled_clf.named_steps['pca']
        pca_std = std_clf.named_steps['pca']

        # Show first principal components
        print('\nPC 1 without scaling:\n', pca.components_[0])
        print('\nPC 1 with scaling:\n', pca_std.components_[0])

        # Use PCA without and with scale on X_train data for visualization.
        X_train_transformed = pca.transform(X_train)
        scaler = std_clf.named_steps['standardscaler']
        X_train_std_transformed = pca_std.transform(scaler.transform(X_train))

        # visualize standardized vs. untouched dataset with PCA performed
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)


        for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
            ax1.scatter(X_train_transformed[y_train == l, 0],
                        X_train_transformed[y_train == l, 1],
                        color=c,
                        label='class %s' % l,
                        alpha=0.5,
                        marker=m
                        )

        for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
            ax2.scatter(X_train_std_transformed[y_train == l, 0],
                        X_train_std_transformed[y_train == l, 1],
                        color=c,
                        label='class %s' % l,
                        alpha=0.5,
                        marker=m
                        )

        ax1.set_title('Training dataset after PCA')
        ax2.set_title('Standardized training dataset after PCA')

        for ax in (ax1, ax2):
            ax.set_xlabel('1st principal component')
            ax.set_ylabel('2nd principal component')
            ax.legend(loc='upper right')
            ax.grid()

        plt.tight_layout()

        plt.show()
    """)
    selenium.run(cmd)