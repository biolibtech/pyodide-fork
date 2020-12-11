from textwrap import dedent
import pytest

def test_sklearn_pipeline_multiple_feature_concat(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from __future__ import print_function
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.datasets import load_iris
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest

        iris = load_iris()

        X, y = iris.data, iris.target

        # This dataset is way too high-dimensional. Better do PCA:
        pca = PCA(n_components=2)

        # Maybe some original features where good, too?
        selection = SelectKBest(k=1)

        # Build estimator from PCA and Univariate selection:

        combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

        # Use combined features to transform dataset:
        X_features = combined_features.fit(X, y).transform(X)
        print("Combined space has", X_features.shape[1], "features")

        svm = SVC(kernel="linear")

        # Do grid search over k, n_components and C:

        pipeline = Pipeline([("features", combined_features), ("svm", svm)])

        param_grid = dict(features__pca__n_components=[1, 2, 3],
                        features__univ_select__k=[1, 2],
                        svm__C=[0.1, 1, 10])

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10)
        grid_search.fit(X, y)
        print(grid_search.best_estimator_)
    """)
    selenium.run(cmd)

def test_sklearn_pipeline_column_transformer_mixed_types(selenium_standalone, request):
    selenium = selenium_standalone
    request.applymarker(pytest.mark.xfail(run=True, reason='Tries to download with urllib'))
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    selenium.load_package("pandas")
    cmd = dedent(r"""
        #downscale_local_mean
        from __future__ import print_function

        import pandas as pd
        import numpy as np

        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, GridSearchCV

        np.random.seed(0)

        # Read data from Titanic dataset.
        titanic_url = ('https://raw.githubusercontent.com/amueller/'
                    'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
        data = pd.read_csv(titanic_url)

        # We will train our classifier with the following features:
        # Numeric Features:
        # - age: float.
        # - fare: float.
        # Categorical Features:
        # - embarked: categories encoded as strings {'C', 'S', 'Q'}.
        # - sex: categories encoded as strings {'female', 'male'}.
        # - pclass: ordinal integers {1, 2, 3}.

        # We create the preprocessing pipelines for both numeric and categorical data.
        numeric_features = ['age', 'fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['embarked', 'sex', 'pclass']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(solver='lbfgs'))])

        X = data.drop('survived', axis=1)
        y = data['survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf.fit(X_train, y_train)
        print("model score: %.3f" % clf.score(X_test, y_test))
    """)
    selenium.run(cmd)

def test_sklearn_pipeline_transforming_targets(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from __future__ import print_function, division

        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        from distutils.version import LooseVersion
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import RidgeCV
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.metrics import median_absolute_error, r2_score


        # `normed` is being deprecated in favor of `density` in histograms
        if LooseVersion(matplotlib.__version__) >= '2.1':
            density_param = {'density': True}
        else:
            density_param = {'normed': True}
        X, y = make_regression(n_samples=10000, noise=100, random_state=0)
        y = np.exp((y + abs(y.min())) / 200)
        y_trans = np.log1p(y)
        f, (ax0, ax1) = plt.subplots(1, 2)

        ax0.hist(y, bins=100, **density_param)
        ax0.set_xlim([0, 2000])
        ax0.set_ylabel('Probability')
        ax0.set_xlabel('Target')
        ax0.set_title('Target distribution')

        ax1.hist(y_trans, bins=100, **density_param)
        ax1.set_ylabel('Probability')
        ax1.set_xlabel('Target')
        ax1.set_title('Transformed target distribution')

        f.suptitle("Synthetic data", y=0.035)
        f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

        regr = RidgeCV()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)

        ax0.scatter(y_test, y_pred)
        ax0.plot([0, 2000], [0, 2000], '--k')
        ax0.set_ylabel('Target predicted')
        ax0.set_xlabel('True Target')
        ax0.set_title('Ridge regression \n without target transformation')
        ax0.text(100, 1750, r'$R^2$=%.2f, MAE=%.2f' % (
            r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
        ax0.set_xlim([0, 2000])
        ax0.set_ylim([0, 2000])

        regr_trans = TransformedTargetRegressor(regressor=RidgeCV(),
                                                func=np.log1p,
                                                inverse_func=np.expm1)
        regr_trans.fit(X_train, y_train)
        y_pred = regr_trans.predict(X_test)

        ax1.scatter(y_test, y_pred)
        ax1.plot([0, 2000], [0, 2000], '--k')
        ax1.set_ylabel('Target predicted')
        ax1.set_xlabel('True Target')
        ax1.set_title('Ridge regression \n with target transformation')
        ax1.text(100, 1750, r'$R^2$=%.2f, MAE=%.2f' % (
            r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
        ax1.set_xlim([0, 2000])
        ax1.set_ylim([0, 2000])

        f.suptitle("Synthetic data", y=0.035)
        f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    """)
    selenium.run(cmd)