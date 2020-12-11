from textwrap import dedent
import pytest

def test_sklearn_real_world_datasets_outlier_detection(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from sklearn.covariance import EllipticEnvelope
        from sklearn.svm import OneClassSVM
        import matplotlib.pyplot as plt
        import matplotlib.font_manager
        from sklearn.datasets import load_boston

        # Get data
        X1 = load_boston()['data'][:, [8, 10]]  # two clusters
        X2 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped

        # Define "classifiers" to be used
        classifiers = {
            "Empirical Covariance": EllipticEnvelope(support_fraction=1.,
                                                    contamination=0.261),
            "Robust Covariance (Minimum Covariance Determinant)":
            EllipticEnvelope(contamination=0.261),
            "OCSVM": OneClassSVM(nu=0.261, gamma=0.05)}
        colors = ['m', 'g', 'b']
        legend1 = {}
        legend2 = {}

        # Learn a frontier for outlier detection with several classifiers
        xx1, yy1 = np.meshgrid(np.linspace(-8, 28, 500), np.linspace(3, 40, 500))
        xx2, yy2 = np.meshgrid(np.linspace(3, 10, 500), np.linspace(-5, 45, 500))
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            plt.figure(1)
            clf.fit(X1)
            Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
            Z1 = Z1.reshape(xx1.shape)
            legend1[clf_name] = plt.contour(
                xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])
            plt.figure(2)
            clf.fit(X2)
            Z2 = clf.decision_function(np.c_[xx2.ravel(), yy2.ravel()])
            Z2 = Z2.reshape(xx2.shape)
            legend2[clf_name] = plt.contour(
                xx2, yy2, Z2, levels=[0], linewidths=2, colors=colors[i])

        legend1_values_list = list(legend1.values())
        legend1_keys_list = list(legend1.keys())

        # Plot the results (= shape of the data points cloud)
        plt.figure(1)  # two clusters
        plt.title("Outlier detection on a real data set (boston housing)")
        plt.scatter(X1[:, 0], X1[:, 1], color='black')
        bbox_args = dict(boxstyle="round", fc="0.8")
        arrow_args = dict(arrowstyle="->")
        plt.annotate("several confounded points", xy=(24, 19),
                    xycoords="data", textcoords="data",
                    xytext=(13, 10), bbox=bbox_args, arrowprops=arrow_args)
        plt.xlim((xx1.min(), xx1.max()))
        plt.ylim((yy1.min(), yy1.max()))
        plt.legend((legend1_values_list[0].collections[0],
                    legend1_values_list[1].collections[0],
                    legend1_values_list[2].collections[0]),
                (legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2]),
                loc="upper center",
                prop=matplotlib.font_manager.FontProperties(size=12))
        plt.ylabel("accessibility to radial highways")
        plt.xlabel("pupil-teacher ratio by town")

        legend2_values_list = list(legend2.values())
        legend2_keys_list = list(legend2.keys())

        plt.figure(2)  # "banana" shape
        plt.title("Outlier detection on a real data set (boston housing)")
        plt.scatter(X2[:, 0], X2[:, 1], color='black')
        plt.xlim((xx2.min(), xx2.max()))
        plt.ylim((yy2.min(), yy2.max()))
        plt.legend((legend2_values_list[0].collections[0],
                    legend2_values_list[1].collections[0],
                    legend2_values_list[2].collections[0]),
                (legend2_keys_list[0], legend2_keys_list[1], legend2_keys_list[2]),
                loc="upper center",
                prop=matplotlib.font_manager.FontProperties(size=12))
        plt.ylabel("% lower status of the population")
        plt.xlabel("average number of rooms per dwelling")

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_real_world_datasets_l1_reconstruction(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from scipy import sparse
        from scipy import ndimage
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import Ridge
        import matplotlib.pyplot as plt


        def _weights(x, dx=1, orig=0):
            x = np.ravel(x)
            floor_x = np.floor((x - orig) / dx).astype(np.int64)
            alpha = (x - orig - floor_x * dx) / dx
            return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


        def _generate_center_coordinates(l_x):
            X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
            center = l_x / 2.
            X += 0.5 - center
            Y += 0.5 - center
            return X, Y


        def build_projection_operator(l_x, n_dir):

            X, Y = _generate_center_coordinates(l_x)
            angles = np.linspace(0, np.pi, n_dir, endpoint=False)
            data_inds, weights, camera_inds = [], [], []
            data_unravel_indices = np.arange(l_x ** 2)
            data_unravel_indices = np.hstack((data_unravel_indices,
                                            data_unravel_indices))
            for i, angle in enumerate(angles):
                Xrot = np.cos(angle) * X - np.sin(angle) * Y
                inds, w = _weights(Xrot, dx=1, orig=X.min())
                mask = np.logical_and(inds >= 0, inds < l_x)
                weights += list(w[mask])
                camera_inds += list(inds[mask] + i * l_x)
                data_inds += list(data_unravel_indices[mask])
            proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
            return proj_operator


        def generate_synthetic_data():
            rs = np.random.RandomState(0)
            n_pts = 36
            x, y = np.ogrid[0:l, 0:l]
            mask_outer = (x - l / 2.) ** 2 + (y - l / 2.) ** 2 < (l / 2.) ** 2
            mask = np.zeros((l, l))
            points = l * rs.rand(2, n_pts)
            mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
            mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
            res = np.logical_and(mask > mask.mean(), mask_outer)
            return np.logical_xor(res, ndimage.binary_erosion(res))


        # Generate synthetic images, and projections
        l = 128
        proj_operator = build_projection_operator(l, l // 7)
        data = generate_synthetic_data()
        proj = proj_operator * data.ravel()[:, np.newaxis]
        proj += 0.15 * np.random.randn(*proj.shape)

        # Reconstruction with L2 (Ridge) penalization
        rgr_ridge = Ridge(alpha=0.2)
        rgr_ridge.fit(proj_operator, proj.ravel())
        rec_l2 = rgr_ridge.coef_.reshape(l, l)

        # Reconstruction with L1 (Lasso) penalization
        # the best value of alpha was determined using cross validation
        # with LassoCV
        rgr_lasso = Lasso(alpha=0.001)
        rgr_lasso.fit(proj_operator, proj.ravel())
        rec_l1 = rgr_lasso.coef_.reshape(l, l)

        plt.figure(figsize=(8, 3.3))
        plt.subplot(131)
        plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
        plt.axis('off')
        plt.title('original image')
        plt.subplot(132)
        plt.imshow(rec_l2, cmap=plt.cm.gray, interpolation='nearest')
        plt.title('L2 penalization')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation='nearest')
        plt.title('L1 penalization')
        plt.axis('off')

        plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                            right=1)

        plt.show()
    """)
    selenium.run(cmd)


def test_sklearn_real_world_datasets_visualize_stock_market(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=True, reason='Tries to fetch data with urllib'))
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    selenium.load_package("pandas")
    cmd = dedent(r"""
        from __future__ import print_function

        # Author: Gael Varoquaux gael.varoquaux@normalesup.org
        # License: BSD 3 clause

        import sys

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        import pandas as pd

        from sklearn import cluster, covariance, manifold

        print(__doc__)


        # #############################################################################
        # Retrieve the data from Internet

        # The data is from 2003 - 2008. This is reasonably calm: (not too long ago so
        # that we get high-tech firms, and before the 2008 crash). This kind of
        # historical data can be obtained for from APIs like the quandl.com and
        # alphavantage.co ones.

        symbol_dict = {
            'TOT': 'Total',
            'XOM': 'Exxon',
            'CVX': 'Chevron',
            'COP': 'ConocoPhillips',
            'VLO': 'Valero Energy',
            'MSFT': 'Microsoft',
            'IBM': 'IBM',
            'TWX': 'Time Warner',
            'CMCSA': 'Comcast',
            'CVC': 'Cablevision',
            'YHOO': 'Yahoo',
            'DELL': 'Dell',
            'HPQ': 'HP',
            'AMZN': 'Amazon',
            'TM': 'Toyota',
            'CAJ': 'Canon',
            'SNE': 'Sony',
            'F': 'Ford',
            'HMC': 'Honda',
            'NAV': 'Navistar',
            'NOC': 'Northrop Grumman',
            'BA': 'Boeing',
            'KO': 'Coca Cola',
            'MMM': '3M',
            'MCD': 'McDonald\'s',
            'PEP': 'Pepsi',
            'K': 'Kellogg',
            'UN': 'Unilever',
            'MAR': 'Marriott',
            'PG': 'Procter Gamble',
            'CL': 'Colgate-Palmolive',
            'GE': 'General Electrics',
            'WFC': 'Wells Fargo',
            'JPM': 'JPMorgan Chase',
            'AIG': 'AIG',
            'AXP': 'American express',
            'BAC': 'Bank of America',
            'GS': 'Goldman Sachs',
            'AAPL': 'Apple',
            'SAP': 'SAP',
            'CSCO': 'Cisco',
            'TXN': 'Texas Instruments',
            'XRX': 'Xerox',
            'WMT': 'Wal-Mart',
            'HD': 'Home Depot',
            'GSK': 'GlaxoSmithKline',
            'PFE': 'Pfizer',
            'SNY': 'Sanofi-Aventis',
            'NVS': 'Novartis',
            'KMB': 'Kimberly-Clark',
            'R': 'Ryder',
            'GD': 'General Dynamics',
            'RTN': 'Raytheon',
            'CVS': 'CVS',
            'CAT': 'Caterpillar',
            'DD': 'DuPont de Nemours'}


        symbols, names = np.array(sorted(symbol_dict.items())).T

        quotes = []

        for symbol in symbols:
            print('Fetching quote history for %r' % symbol, file=sys.stderr)
            url = ('https://raw.githubusercontent.com/scikit-learn/examples-data/'
                'master/financial-data/{}.csv')
            quotes.append(pd.read_csv(url.format(symbol)))

        close_prices = np.vstack([q['close'] for q in quotes])
        open_prices = np.vstack([q['open'] for q in quotes])

        # The daily variations of the quotes are what carry most information
        variation = close_prices - open_prices


        # #############################################################################
        # Learn a graphical structure from the correlations
        edge_model = covariance.GraphicalLassoCV(cv=5)

        # standardize the time series: using correlations rather than covariance
        # is more efficient for structure recovery
        X = variation.copy().T
        X /= X.std(axis=0)
        edge_model.fit(X)

        # #############################################################################
        # Cluster using affinity propagation

        _, labels = cluster.affinity_propagation(edge_model.covariance_)
        n_labels = labels.max()

        for i in range(n_labels + 1):
            print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

        # #############################################################################
        # Find a low-dimension embedding for visualization: find the best position of
        # the nodes (the stocks) on a 2D plane

        # We use a dense eigen_solver to achieve reproducibility (arpack is
        # initiated with random vectors that we don't control). In addition, we
        # use a large number of neighbors to capture the large-scale structure.
        node_position_model = manifold.LocallyLinearEmbedding(
            n_components=2, eigen_solver='dense', n_neighbors=6)

        embedding = node_position_model.fit_transform(X.T).T

        # #############################################################################
        # Visualization
        plt.figure(1, facecolor='w', figsize=(10, 8))
        plt.clf()
        ax = plt.axes([0., 0., 1., 1.])
        plt.axis('off')

        # Display a graph of the partial correlations
        partial_correlations = edge_model.precision_.copy()
        d = 1 / np.sqrt(np.diag(partial_correlations))
        partial_correlations *= d
        partial_correlations *= d[:, np.newaxis]
        non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

        # Plot the nodes using the coordinates of our embedding
        plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                    cmap=plt.cm.nipy_spectral)

        # Plot the edges
        start_idx, end_idx = np.where(non_zero)
        # a sequence of (*line0*, *line1*, *line2*), where::
        #            linen = (x0, y0), (x1, y1), ... (xm, ym)
        segments = [[embedding[:, start], embedding[:, stop]]
                    for start, stop in zip(start_idx, end_idx)]
        values = np.abs(partial_correlations[non_zero])
        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.hot_r,
                            norm=plt.Normalize(0, .7 * values.max()))
        lc.set_array(values)
        lc.set_linewidths(15 * values)
        ax.add_collection(lc)

        # Add a label to each node. The challenge here is that we want to
        # position the labels to avoid overlap with other labels
        for index, (name, label, (x, y)) in enumerate(
                zip(names, labels, embedding.T)):

            dx = x - embedding[0]
            dx[index] = 1
            dy = y - embedding[1]
            dy[index] = 1
            this_dx = dx[np.argmin(np.abs(dy))]
            this_dy = dy[np.argmin(np.abs(dx))]
            if this_dx > 0:
                horizontalalignment = 'left'
                x = x + .002
            else:
                horizontalalignment = 'right'
                x = x - .002
            if this_dy > 0:
                verticalalignment = 'bottom'
                y = y + .002
            else:
                verticalalignment = 'top'
                y = y - .002
            plt.text(x, y, name, size=10,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    bbox=dict(facecolor='w',
                            edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                            alpha=.6))

        plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
                embedding[0].max() + .10 * embedding[0].ptp(),)
        plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
                embedding[1].max() + .03 * embedding[1].ptp())

        plt.show()
    """)
    selenium.run(cmd)