from textwrap import dedent
import pytest

def test_sklearn_calibration_compare_calibrate(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        np.random.seed(0)

        import matplotlib.pyplot as plt

        from sklearn import datasets
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        from sklearn.calibration import calibration_curve

        X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                            n_informative=2, n_redundant=2)

        train_samples = 100  # Samples used for training the models

        X_train = X[:train_samples]
        X_test = X[train_samples:]
        y_train = y[:train_samples]
        y_test = y[train_samples:]

        # Create classifiers
        lr = LogisticRegression(solver='lbfgs')
        gnb = GaussianNB()
        svc = LinearSVC(C=1.0)
        rfc = RandomForestClassifier(n_estimators=100)


        # #############################################################################
        # Plot calibration plots

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for clf, name in [(lr, 'Logistic'),
                        (gnb, 'Naive Bayes'),
                        (svc, 'Support Vector Classification'),
                        (rfc, 'Random Forest')]:
            clf.fit(X_train, y_train)
            if hasattr(clf, "predict_proba"):
                prob_pos = clf.predict_proba(X_test)[:, 1]
            else:  # use decision function
                prob_pos = clf.decision_function(X_test)
                prob_pos = \
                    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, prob_pos, n_bins=10)

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label="%s" % (name, ))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                    histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_calibration_plot_calibration_curve(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        #downscale_local_mean
        import matplotlib.pyplot as plt

        from sklearn import datasets
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                                    f1_score)
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve
        from sklearn.model_selection import train_test_split


        # Create dataset of classification task with many redundant and few
        # informative features
        X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                            n_informative=2, n_redundant=10,
                                            random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
                                                            random_state=42)


        def plot_calibration_curve(est, name, fig_index):

            # Calibrated with isotonic calibration
            isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

            # Calibrated with sigmoid calibration
            sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

            # Logistic regression with no calibration as baseline
            lr = LogisticRegression(C=1., solver='lbfgs')

            fig = plt.figure(fig_index, figsize=(10, 10))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))

            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            for clf, name in [(lr, 'Logistic'),
                            (est, name),
                            (isotonic, name + ' + Isotonic'),
                            (sigmoid, name + ' + Sigmoid')]:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if hasattr(clf, "predict_proba"):
                    prob_pos = clf.predict_proba(X_test)[:, 1]
                else:  # use decision function
                    prob_pos = clf.decision_function(X_test)
                    prob_pos = \
                        (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

                clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
                print("%s:" % name)
                print("\tBrier: %1.3f" % (clf_score))
                print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
                print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
                print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

                fraction_of_positives, mean_predicted_value = \
                    calibration_curve(y_test, prob_pos, n_bins=10)

                ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                        label="%s (%1.3f)" % (name, clf_score))

                ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                        histtype="step", lw=2)

            ax1.set_ylabel("Fraction of positives")
            ax1.set_ylim([-0.05, 1.05])
            ax1.legend(loc="lower right")
            ax1.set_title('Calibration plots  (reliability curve)')

            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Count")
            ax2.legend(loc="upper center", ncol=2)

            plt.tight_layout()

        # Plot calibration curve for Gaussian Naive Bayes
        plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

        # Plot calibration curve for Linear SVC
        plot_calibration_curve(LinearSVC(), "SVC", 2)

        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_calibration_probability_calibration_of_classifiers(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm

        from sklearn.datasets import make_blobs
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import brier_score_loss
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import train_test_split


        n_samples = 50000
        n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

        # Generate 3 blobs with 2 classes where the second blob contains
        # half positive samples and half negative samples. Probability in this
        # blob is therefore 0.5.
        centers = [(-5, -5), (0, 0), (5, 5)]
        X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                        centers=centers, shuffle=False, random_state=42)

        y[:n_samples // 2] = 0
        y[n_samples // 2:] = 1
        sample_weight = np.random.RandomState(42).rand(y.shape[0])

        # split train, test for calibration
        X_train, X_test, y_train, y_test, sw_train, sw_test = \
            train_test_split(X, y, sample_weight, test_size=0.9, random_state=42)

        # Gaussian Naive-Bayes with no calibration
        clf = GaussianNB()
        clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
        prob_pos_clf = clf.predict_proba(X_test)[:, 1]

        # Gaussian Naive-Bayes with isotonic calibration
        clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
        clf_isotonic.fit(X_train, y_train, sw_train)
        prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

        # Gaussian Naive-Bayes with sigmoid calibration
        clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
        clf_sigmoid.fit(X_train, y_train, sw_train)
        prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

        print("Brier scores: (the smaller the better)")

        clf_score = brier_score_loss(y_test, prob_pos_clf, sw_test)
        print("No calibration: %1.3f" % clf_score)

        clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sw_test)
        print("With isotonic calibration: %1.3f" % clf_isotonic_score)

        clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sw_test)
        print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

        # #############################################################################
        # Plot the data and the predicted probabilities
        plt.figure()
        y_unique = np.unique(y)
        colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
        for this_y, color in zip(y_unique, colors):
            this_X = X_train[y_train == this_y]
            this_sw = sw_train[y_train == this_y]
            plt.scatter(this_X[:, 0], this_X[:, 1], s=this_sw * 50, c=color,
                        alpha=0.5, edgecolor='k',
                        label="Class %s" % this_y)
        plt.legend(loc="best")
        plt.title("Data")

        plt.figure()
        order = np.lexsort((prob_pos_clf, ))
        plt.plot(prob_pos_clf[order], 'r', label='No calibration (%1.3f)' % clf_score)
        plt.plot(prob_pos_isotonic[order], 'g', linewidth=3,
                label='Isotonic calibration (%1.3f)' % clf_isotonic_score)
        plt.plot(prob_pos_sigmoid[order], 'b', linewidth=3,
                label='Sigmoid calibration (%1.3f)' % clf_sigmoid_score)
        plt.plot(np.linspace(0, y_test.size, 51)[1::2],
                y_test[order].reshape(25, -1).mean(1),
                'k', linewidth=3, label=r'Empirical')
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Instances sorted according to predicted probability "
                "(uncalibrated GNB)")
        plt.ylabel("P(y=1)")
        plt.legend(loc="upper left")
        plt.title("Gaussian naive Bayes probabilities")

        plt.show()
    """)
    selenium.run(cmd)