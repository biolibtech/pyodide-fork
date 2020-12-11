from textwrap import dedent
import pytest

def test_sklearn_feature_selection_recursive_feature_estimation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from sklearn.svm import SVC
        from sklearn.datasets import load_digits
        from sklearn.feature_selection import RFE
        import matplotlib.pyplot as plt

        # Load the digits dataset
        digits = load_digits()
        X = digits.images.reshape((len(digits.images), -1))
        y = digits.target

        # Create the RFE object and rank each pixel
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
        rfe.fit(X, y)
        ranking = rfe.ranking_.reshape(digits.images[0].shape)

        # Plot pixel ranking
        plt.matshow(ranking, cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title("Ranking of pixels with RFE")
        plt.show()
    """)
    selenium.run(cmd)


def test_sklearn_feature_selection_f_test(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.feature_selection import f_regression, mutual_info_regression

        np.random.seed(0)
        X = np.random.rand(1000, 3)
        y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

        f_test, _ = f_regression(X, y)
        f_test /= np.max(f_test)

        mi = mutual_info_regression(X, y)
        mi /= np.max(mi)

        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.scatter(X[:, i], y, edgecolor='black', s=20)
            plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
            if i == 0:
                plt.ylabel("$y$", fontsize=14)
            plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
                    fontsize=16)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_feature_selection_recursive_feature_estimation_with_cross_validation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import matplotlib.pyplot as plt
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold
        from sklearn.feature_selection import RFECV
        from sklearn.datasets import make_classification

        # Build a classification task using 3 informative features
        X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                                n_redundant=2, n_repeated=0, n_classes=8,
                                n_clusters_per_class=1, random_state=0)

        # Create the RFE object and compute a cross-validated score.
        svc = SVC(kernel="linear")
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                    scoring='accuracy')
        rfecv.fit(X, y)

        print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_feature_selection_feature_selection_with_lasso_cv(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""

        import matplotlib.pyplot as plt
        import numpy as np

        from sklearn.datasets import load_boston
        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import LassoCV

        # Load the boston dataset.
        boston = load_boston()
        X, y = boston['data'], boston['target']

        # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
        clf = LassoCV(cv=5)

        # Set a minimum threshold of 0.25
        sfm = SelectFromModel(clf, threshold=0.25)
        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]

        # Reset the threshold till the number of features equals two.
        # Note that the attribute can be set directly instead of repeatedly
        # fitting the metatransformer.
        while n_features > 2:
            sfm.threshold += 0.1
            X_transform = sfm.transform(X)
            n_features = X_transform.shape[1]

        # Plot the selected two features from X.
        plt.title(
            "Features selected from Boston using SelectFromModel with "
            "threshold %0.3f." % sfm.threshold)
        feature1 = X_transform[:, 0]
        feature2 = X_transform[:, 1]
        plt.plot(feature1, feature2, 'r.')
        plt.xlabel("Feature number 1")
        plt.ylabel("Feature number 2")
        plt.ylim([np.min(feature2), np.max(feature2)])
        plt.show()
    """)
    selenium.run(cmd)


def test_sklearn_feature_selection_univariate_feature_selection(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import datasets, svm
        from sklearn.feature_selection import SelectPercentile, f_classif

        # #############################################################################
        # Import some data to play with

        # The iris dataset
        iris = datasets.load_iris()

        # Some noisy data not correlated
        E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

        # Add the noisy data to the informative features
        X = np.hstack((iris.data, E))
        y = iris.target

        plt.figure(1)
        plt.clf()

        X_indices = np.arange(X.shape[-1])

        # #############################################################################
        # Univariate feature selection with F-test for feature scoring
        # We use the default selection function: the 10% most significant features
        selector = SelectPercentile(f_classif, percentile=10)
        selector.fit(X, y)
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        plt.bar(X_indices - .45, scores, width=.2,
                label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
                edgecolor='black')

        # #############################################################################
        # Compare to the weights of an SVM
        clf = svm.SVC(kernel='linear')
        clf.fit(X, y)

        svm_weights = (clf.coef_ ** 2).sum(axis=0)
        svm_weights /= svm_weights.max()

        plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
                color='navy', edgecolor='black')

        clf_selected = svm.SVC(kernel='linear')
        clf_selected.fit(selector.transform(X), y)

        svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
        svm_weights_selected /= svm_weights_selected.max()

        plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
                width=.2, label='SVM weights after selection', color='c',
                edgecolor='black')


        plt.title("Comparing feature selection")
        plt.xlabel('Feature number')
        plt.yticks(())
        plt.axis('tight')
        plt.legend(loc='upper right')
        plt.show()
    """)
    selenium.run(cmd)

    