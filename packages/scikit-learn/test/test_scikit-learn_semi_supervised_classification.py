from textwrap import dedent
import pytest

def test_sklearn_semi_supervised_classification_decision_boundary_vs_SVM(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import datasets
        from sklearn import svm
        from sklearn.semi_supervised import label_propagation

        rng = np.random.RandomState(0)

        iris = datasets.load_iris()

        X = iris.data[:, :2]
        y = iris.target

        # step size in the mesh
        h = .02

        y_30 = np.copy(y)
        y_30[rng.rand(len(y)) < 0.3] = -1
        y_50 = np.copy(y)
        y_50[rng.rand(len(y)) < 0.5] = -1
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        ls30 = (label_propagation.LabelSpreading().fit(X, y_30),
                y_30)
        ls50 = (label_propagation.LabelSpreading().fit(X, y_50),
                y_50)
        ls100 = (label_propagation.LabelSpreading().fit(X, y), y)
        rbf_svc = (svm.SVC(kernel='rbf', gamma=.5).fit(X, y), y)

        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # title for the plots
        titles = ['Label Spreading 30% data',
                'Label Spreading 50% data',
                'Label Spreading 100% data',
                'SVC with rbf kernel']

        color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}

        for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(2, 2, i + 1)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
            plt.axis('off')

            # Plot also the training points
            colors = [color_map[y] for y in y_train]
            plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')

            plt.title(titles[i])

        plt.suptitle("Unlabeled points are colored white", y=0.1)
        plt.show()

    """)
    selenium.run(cmd)

def test_sklearn_semi_supervised_classification_label_propagation(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.semi_supervised import label_propagation
        from sklearn.datasets import make_circles

        # generate ring with inner box
        n_samples = 200
        X, y = make_circles(n_samples=n_samples, shuffle=False)
        outer, inner = 0, 1
        labels = np.full(n_samples, -1.)
        labels[0] = outer
        labels[-1] = inner

        # #############################################################################
        # Learn with LabelSpreading
        label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=0.8)
        label_spread.fit(X, labels)

        # #############################################################################
        # Plot output labels
        output_labels = label_spread.transduction_
        plt.figure(figsize=(8.5, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color='navy',
                    marker='s', lw=0, label="outer labeled", s=10)
        plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color='c',
                    marker='s', lw=0, label='inner labeled', s=10)
        plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange',
                    marker='.', label='unlabeled')
        plt.legend(scatterpoints=1, shadow=False, loc='upper right')
        plt.title("Raw data (2 classes=outer and inner)")

        plt.subplot(1, 2, 2)
        output_label_array = np.asarray(output_labels)
        outer_numbers = np.where(output_label_array == outer)[0]
        inner_numbers = np.where(output_label_array == inner)[0]
        plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',
                    marker='s', lw=0, s=10, label="outer learned")
        plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',
                    marker='s', lw=0, s=10, label="inner learned")
        plt.legend(scatterpoints=1, shadow=False, loc='upper right')
        plt.title("Labels learned with Label Spreading (KNN)")

        plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_semi_supervised_classification_label_propagation_digits(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt

        from scipy import stats

        from sklearn import datasets
        from sklearn.semi_supervised import label_propagation

        from sklearn.metrics import confusion_matrix, classification_report

        digits = datasets.load_digits()
        rng = np.random.RandomState(0)
        indices = np.arange(len(digits.data))
        rng.shuffle(indices)

        X = digits.data[indices[:330]]
        y = digits.target[indices[:330]]
        images = digits.images[indices[:330]]

        n_total_samples = len(y)
        n_labeled_points = 30

        indices = np.arange(n_total_samples)

        unlabeled_set = indices[n_labeled_points:]

        # #############################################################################
        # Shuffle everything around
        y_train = np.copy(y)
        y_train[unlabeled_set] = -1

        # #############################################################################
        # Learn with LabelSpreading
        lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
        lp_model.fit(X, y_train)
        predicted_labels = lp_model.transduction_[unlabeled_set]
        true_labels = y[unlabeled_set]

        cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

        print("Label Spreading model: %d labeled & %d unlabeled points (%d total)" %
            (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))

        print(classification_report(true_labels, predicted_labels))

        print("Confusion matrix")
        print(cm)

        # #############################################################################
        # Calculate uncertainty values for each transduced distribution
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

        # #############################################################################
        # Pick the top 10 most uncertain labels
        uncertainty_index = np.argsort(pred_entropies)[-10:]

        # #############################################################################
        # Plot
        f = plt.figure(figsize=(7, 5))
        for index, image_index in enumerate(uncertainty_index):
            image = images[image_index]

            sub = f.add_subplot(2, 5, index + 1)
            sub.imshow(image, cmap=plt.cm.gray_r)
            plt.xticks([])
            plt.yticks([])
            sub.set_title('predict: %i\ntrue: %i' % (
                lp_model.transduction_[image_index], y[image_index]))

        f.suptitle('Learning with small amount of labeled data')
        plt.show()
    """)
    selenium.run(cmd)

def test_sklearn_semi_supervised_classification_label_propagation_active_learning(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats

        from sklearn import datasets
        from sklearn.semi_supervised import label_propagation
        from sklearn.metrics import classification_report, confusion_matrix

        digits = datasets.load_digits()
        rng = np.random.RandomState(0)
        indices = np.arange(len(digits.data))
        rng.shuffle(indices)

        X = digits.data[indices[:330]]
        y = digits.target[indices[:330]]
        images = digits.images[indices[:330]]

        n_total_samples = len(y)
        n_labeled_points = 10
        max_iterations = 5

        unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
        f = plt.figure()

        for i in range(max_iterations):
            if len(unlabeled_indices) == 0:
                print("No unlabeled items left to label.")
                break
            y_train = np.copy(y)
            y_train[unlabeled_indices] = -1

            lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
            lp_model.fit(X, y_train)

            predicted_labels = lp_model.transduction_[unlabeled_indices]
            true_labels = y[unlabeled_indices]

            cm = confusion_matrix(true_labels, predicted_labels,
                                labels=lp_model.classes_)

            print("Iteration %i %s" % (i, 70 * "_"))
            print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
                % (n_labeled_points, n_total_samples - n_labeled_points,
                    n_total_samples))

            print(classification_report(true_labels, predicted_labels))

            print("Confusion matrix")
            print(cm)

            # compute the entropies of transduced label distributions
            pred_entropies = stats.distributions.entropy(
                lp_model.label_distributions_.T)

            # select up to 5 digit examples that the classifier is most uncertain about
            uncertainty_index = np.argsort(pred_entropies)[::-1]
            uncertainty_index = uncertainty_index[
                np.in1d(uncertainty_index, unlabeled_indices)][:5]

            # keep track of indices that we get labels for
            delete_indices = np.array([])

            # for more than 5 iterations, visualize the gain only on the first 5
            if i < 5:
                f.text(.05, (1 - (i + 1) * .183),
                    "model %d\n\nfit with\n%d labels" %
                    ((i + 1), i * 5 + 10), size=10)
            for index, image_index in enumerate(uncertainty_index):
                image = images[image_index]

                # for more than 5 iterations, visualize the gain only on the first 5
                if i < 5:
                    sub = f.add_subplot(5, 5, index + 1 + (5 * i))
                    sub.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
                    sub.set_title("predict: %i\ntrue: %i" % (
                        lp_model.transduction_[image_index], y[image_index]), size=10)
                    sub.axis('off')

                # labeling 5 points, remote from labeled set
                delete_index, = np.where(unlabeled_indices == image_index)
                delete_indices = np.concatenate((delete_indices, delete_index))

            unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
            n_labeled_points += len(uncertainty_index)

        f.suptitle("Active learning with Label Propagation.\nRows show 5 most "
                "uncertain labels to learn with the next model.", y=1.15)
        plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2,
                            hspace=0.85)
        plt.show()
    """)
    selenium.run(cmd)