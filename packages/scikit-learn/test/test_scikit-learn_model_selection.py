from textwrap import dedent
import pytest

def test_sklearn_model_selection_underfitting_vs_overfitting(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score


        def true_fun(X):
            return np.cos(1.5 * np.pi * X)

        np.random.seed(0)

        n_samples = 30
        degrees = [1, 4, 15]

        X = np.sort(np.random.rand(n_samples))
        y = true_fun(X) + np.random.randn(n_samples) * 0.1

        plt.figure(figsize=(14, 5))
        for i in range(len(degrees)):
            ax = plt.subplot(1, len(degrees), i + 1)
            plt.setp(ax, xticks=(), yticks=())

            polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                    include_bias=False)
            linear_regression = LinearRegression()
            pipeline = Pipeline([("polynomial_features", polynomial_features),
                                ("linear_regression", linear_regression)])
            pipeline.fit(X[:, np.newaxis], y)

            # Evaluate the models using crossvalidation
            scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                                    scoring="neg_mean_squared_error", cv=10)

            X_test = np.linspace(0, 1, 100)
            plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
            plt.plot(X_test, true_fun(X_test), label="True function")
            plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim((0, 1))
            plt.ylim((-2, 2))
            plt.legend(loc="best")
            plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
                degrees[i], -scores.mean(), scores.std()))
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_model_selection_nested_versus_non_nested_CV(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from sklearn.datasets import load_iris
        from matplotlib import pyplot as plt
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
        import numpy as np

        print(__doc__)

        # Number of random trials
        NUM_TRIALS = 30

        # Load the dataset
        iris = load_iris()
        X_iris = iris.data
        y_iris = iris.target

        # Set up possible values of parameters to optimize over
        p_grid = {"C": [1, 10, 100],
                "gamma": [.01, .1]}

        # We will use a Support Vector Classifier with "rbf" kernel
        svm = SVC(kernel="rbf")

        # Arrays to store scores
        non_nested_scores = np.zeros(NUM_TRIALS)
        nested_scores = np.zeros(NUM_TRIALS)

        # Loop for each trial
        for i in range(NUM_TRIALS):

            # Choose cross-validation techniques for the inner and outer loops,
            # independently of the dataset.
            # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
            inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
            outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

            # Non_nested parameter search and scoring
            clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
            clf.fit(X_iris, y_iris)
            non_nested_scores[i] = clf.best_score_

            # Nested CV with parameter optimization
            nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
            nested_scores[i] = nested_score.mean()

        score_difference = non_nested_scores - nested_scores

        print("Average difference of {0:6f} with std. dev. of {1:6f}."
            .format(score_difference.mean(), score_difference.std()))

        # Plot scores on each trial for nested and non-nested CV
        plt.figure()
        plt.subplot(211)
        non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
        nested_line, = plt.plot(nested_scores, color='b')
        plt.ylabel("score", fontsize="14")
        plt.legend([non_nested_scores_line, nested_line],
                ["Non-Nested CV", "Nested CV"],
                bbox_to_anchor=(0, .4, .5, 0))
        plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
                x=.5, y=1.1, fontsize="15")

        # Plot bar chart of the difference.
        plt.subplot(212)
        difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
        plt.xlabel("Individual Trial #")
        plt.legend([difference_plot],
                ["Non-Nested CV - Nested CV Score"],
                bbox_to_anchor=(0, 1, .8, 0))
        plt.ylabel("score difference", fontsize="14")

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_model_selection_ROC_with_CV(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        from scipy import interp
        import matplotlib.pyplot as plt

        from sklearn import svm, datasets
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import StratifiedKFold

        # #############################################################################
        # Data IO and generation

        # Import some data to play with
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X, y = X[y != 2], y[y != 2]
        n_samples, n_features = X.shape

        # Add noisy features
        random_state = np.random.RandomState(0)
        X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

        # #############################################################################
        # Classification and ROC analysis

        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=6)
        classifier = svm.SVC(kernel='linear', probability=True,
                            random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(X, y):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    """)
    selenium.run(cmd)

def test_sklearn_model_selection_confusion_matrix(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import svm, datasets
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels

        # import some data to play with
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        class_names = iris.target_names

        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        # Run classifier, using a model that is too regularized (C too low) to see
        # the impact on the results
        classifier = svm.SVC(kernel='linear', C=0.01)
        y_pred = classifier.fit(X_train, y_train).predict(X_test)


        def plot_confusion_matrix(y_true, y_pred, classes,
                                normalize=False,
                                title=None,
                                cmap=plt.cm.Blues):
            if not title:
                if normalize:
                    title = 'Normalized confusion matrix'
                else:
                    title = 'Confusion matrix, without normalization'

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            # Only use the labels that appear in the data
            classes = classes[unique_labels(y_true, y_pred)]
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                # ... and label them with the respective list entries
                xticklabels=classes, yticklabels=classes,
                title=title,
                ylabel='True label',
                xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            return ax


        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, classes=class_names,
                            title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_model_selection_visualizing_CV(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                            StratifiedKFold, GroupShuffleSplit,
                                            GroupKFold, StratifiedShuffleSplit)
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        np.random.seed(1338)
        cmap_data = plt.cm.Paired
        cmap_cv = plt.cm.coolwarm
        n_splits = 4
        # Generate the class/group data
        n_points = 100
        X = np.random.randn(100, 10)

        percentiles_classes = [.1, .3, .6]
        y = np.hstack([[ii] * int(100 * perc)
                    for ii, perc in enumerate(percentiles_classes)])

        # Evenly spaced groups repeated once
        groups = np.hstack([[ii] * 10 for ii in range(10)])


        def visualize_groups(classes, groups, name):
            # Visualize dataset groups
            fig, ax = plt.subplots()
            ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
                    lw=50, cmap=cmap_data)
            ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
                    lw=50, cmap=cmap_data)
            ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
                yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Sample index")


        visualize_groups(y, groups, 'no groups')

        def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):

            # Generate the training/testing visualizations for each CV split
            for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
                # Fill in indices with the training/test groups
                indices = np.array([np.nan] * len(X))
                indices[tt] = 1
                indices[tr] = 0

                # Visualize the results
                ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                        c=indices, marker='_', lw=lw, cmap=cmap_cv,
                        vmin=-.2, vmax=1.2)

            # Plot the data classes and groups at the end
            ax.scatter(range(len(X)), [ii + 1.5] * len(X),
                    c=y, marker='_', lw=lw, cmap=cmap_data)

            ax.scatter(range(len(X)), [ii + 2.5] * len(X),
                    c=group, marker='_', lw=lw, cmap=cmap_data)

            # Formatting
            yticklabels = list(range(n_splits)) + ['class', 'group']
            ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
                xlabel='Sample index', ylabel="CV iteration",
                ylim=[n_splits+2.2, -.2], xlim=[0, 100])
            ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
            return ax

        fig, ax = plt.subplots()
        cv = KFold(n_splits)
        plot_cv_indices(cv, X, y, groups, ax, n_splits)

        fig, ax = plt.subplots()
        cv = StratifiedKFold(n_splits)
        plot_cv_indices(cv, X, y, groups, ax, n_splits)

        cvs = [KFold, GroupKFold, ShuffleSplit, StratifiedKFold,
        GroupShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit]


        for cv in cvs:
            this_cv = cv(n_splits=n_splits)
            fig, ax = plt.subplots(figsize=(6, 3))
            plot_cv_indices(this_cv, X, y, groups, ax, n_splits)

            ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
                    ['Testing set', 'Training set'], loc=(1.02, .8))
            # Make the legend fit
            plt.tight_layout()
            fig.subplots_adjust(right=.7)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_model_selection_(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from __future__ import print_function
        from sklearn import svm, datasets
        from sklearn.model_selection import train_test_split
        import numpy as np

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        # Add noisy features
        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

        # Limit to the two first classes, and split into training and test
        X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                            test_size=.5,
                                                            random_state=random_state)

        # Create a simple classifier
        classifier = svm.LinearSVC(random_state=random_state)
        classifier.fit(X_train, y_train)
        y_score = classifier.decision_function(X_test)

        from sklearn.metrics import average_precision_score
        average_precision = average_precision_score(y_test, y_score)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        from sklearn.metrics import precision_recall_curve
        import matplotlib.pyplot as plt
        from sklearn.utils.fixes import signature

        precision, recall, _ = precision_recall_curve(y_test, y_score)

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                    if 'step' in signature(plt.fill_between).parameters
                    else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
                average_precision))

        from sklearn.preprocessing import label_binarize

        # Use label_binarize to be multi-label like settings
        Y = label_binarize(y, classes=[0, 1, 2])
        n_classes = Y.shape[1]

        # Split into training and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                            random_state=random_state)

        # We use OneVsRestClassifier for multi-label prediction
        from sklearn.multiclass import OneVsRestClassifier

        # Run classifier
        classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
        classifier.fit(X_train, Y_train)
        y_score = classifier.decision_function(X_test)

        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score

        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
            y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                            average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
            .format(average_precision["micro"]))

        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                where='post')
        plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                        **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))

        from itertools import cycle
        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                    ''.format(average_precision["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                        ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

        plt.show()

    """)
    selenium.run(cmd)
