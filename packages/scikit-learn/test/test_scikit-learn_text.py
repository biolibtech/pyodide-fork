from textwrap import dedent
import pytest

def test_sklearn_text_k_means_clustering(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        from sklearn import metrics

        from sklearn.cluster import KMeans, MiniBatchKMeans

        import logging
        from optparse import OptionParser
        import sys
        from time import time

        import numpy as np


        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        # parse commandline arguments
        op = OptionParser()
        op.add_option("--lsa",
                    dest="n_components", type="int",
                    help="Preprocess documents with latent semantic analysis.")
        op.add_option("--no-minibatch",
                    action="store_false", dest="minibatch", default=True,
                    help="Use ordinary k-means algorithm (in batch mode).")
        op.add_option("--no-idf",
                    action="store_false", dest="use_idf", default=True,
                    help="Disable Inverse Document Frequency feature weighting.")
        op.add_option("--use-hashing",
                    action="store_true", default=False,
                    help="Use a hashing feature vectorizer")
        op.add_option("--n-features", type=int, default=10000,
                    help="Maximum number of features (dimensions)"
                        " to extract from text.")
        op.add_option("--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="Print progress reports inside k-means algorithm.")

        print(__doc__)
        op.print_help()


        def is_interactive():
            return not hasattr(sys.modules['__main__'], '__file__')


        # work-around for Jupyter notebook and IPython console
        argv = [] if is_interactive() else sys.argv[1:]
        (opts, args) = op.parse_args(argv)
        if len(args) > 0:
            op.error("this script takes no arguments.")
            sys.exit(1)


        # #############################################################################
        # Load some categories from the training set
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]
        # Uncomment the following to do the analysis on all the categories
        # categories = None

        print("Loading 20 newsgroups dataset for categories:")
        print(categories)

        dataset = fetch_20newsgroups(subset='all', categories=categories,
                                    shuffle=True, random_state=42)

        print("%d documents" % len(dataset.data))
        print("%d categories" % len(dataset.target_names))
        print()

        labels = dataset.target
        true_k = np.unique(labels).shape[0]

        print("Extracting features from the training dataset "
            "using a sparse vectorizer")
        t0 = time()
        if opts.use_hashing:
            if opts.use_idf:
                # Perform an IDF normalization on the output of HashingVectorizer
                hasher = HashingVectorizer(n_features=opts.n_features,
                                        stop_words='english', alternate_sign=False,
                                        norm=None, binary=False)
                vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                vectorizer = HashingVectorizer(n_features=opts.n_features,
                                            stop_words='english',
                                            alternate_sign=False, norm='l2',
                                            binary=False)
        else:
            vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                        min_df=2, stop_words='english',
                                        use_idf=opts.use_idf)
        X = vectorizer.fit_transform(dataset.data)

        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % X.shape)
        print()

        if opts.n_components:
            print("Performing dimensionality reduction using LSA")
            t0 = time()
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            svd = TruncatedSVD(opts.n_components)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)

            X = lsa.fit_transform(X)

            print("done in %fs" % (time() - t0))

            explained_variance = svd.explained_variance_ratio_.sum()
            print("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))

            print()


        # #############################################################################
        # Do the actual clustering

        if opts.minibatch:
            km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                init_size=1000, batch_size=1000, verbose=opts.verbose)
        else:
            km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                        verbose=opts.verbose)

        print("Clustering sparse data with %s" % km)
        t0 = time()
        km.fit(X)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
        print("Adjusted Rand-Index: %.3f"
            % metrics.adjusted_rand_score(labels, km.labels_))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, km.labels_, sample_size=1000))

        print()


        if not opts.use_hashing:
            print("Top terms per cluster:")

            if opts.n_components:
                original_space_centroids = svd.inverse_transform(km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]

            terms = vectorizer.get_feature_names()
            for i in range(true_k):
                print("Cluster %d:" % i, end='')
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind], end='')
                print()
    """)
    selenium.run(cmd)

def test_sklearn_text_feature_hasher_and_dict_vectorizer(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        from __future__ import print_function
        from collections import defaultdict
        import re
        import sys
        from time import time

        import numpy as np

        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction import DictVectorizer, FeatureHasher


        def n_nonzero_columns(X):
            return len(np.unique(X.nonzero()[1]))


        def tokens(doc):

            return (tok.lower() for tok in re.findall(r"\w+", doc))


        def token_freqs(doc):
            freq = defaultdict(int)
            for tok in tokens(doc):
                freq[tok] += 1
            return freq


        categories = [
            'alt.atheism',
            'comp.graphics',
            'comp.sys.ibm.pc.hardware',
            'misc.forsale',
            'rec.autos',
            'sci.space',
            'talk.religion.misc',
        ]
        # Uncomment the following line to use a larger set (11k+ documents)
        # categories = None

        print(__doc__)
        print("Usage: %s [n_features_for_hashing]" % sys.argv[0])
        print("    The default number of features is 2**18.")
        print()

        try:
            n_features = int(sys.argv[1])
        except IndexError:
            n_features = 2 ** 18
        except ValueError:
            print("not a valid number of features: %r" % sys.argv[1])
            sys.exit(1)


        print("Loading 20 newsgroups training data")
        raw_data = fetch_20newsgroups(subset='train', categories=categories).data
        data_size_mb = sum(len(s.encode('utf-8')) for s in raw_data) / 1e6
        print("%d documents - %0.3fMB" % (len(raw_data), data_size_mb))
        print()

        print("DictVectorizer")
        t0 = time()
        vectorizer = DictVectorizer()
        vectorizer.fit_transform(token_freqs(d) for d in raw_data)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
        print("Found %d unique terms" % len(vectorizer.get_feature_names()))
        print()

        print("FeatureHasher on frequency dicts")
        t0 = time()
        hasher = FeatureHasher(n_features=n_features)
        X = hasher.transform(token_freqs(d) for d in raw_data)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
        print("Found %d unique terms" % n_nonzero_columns(X))
        print()

        print("FeatureHasher on raw tokens")
        t0 = time()
        hasher = FeatureHasher(n_features=n_features, input_type="string")
        X = hasher.transform(tokens(d) for d in raw_data)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
        print("Found %d unique terms" % n_nonzero_columns(X))
    """)
    selenium.run(cmd)

def test_sklearn_text_sparse_feature_classification(selenium_standalone, request):
    selenium = selenium_standalone
    selenium.load_package("scikit-learn")
    selenium.load_package("matplotlib")
    cmd = dedent(r"""
        import logging
        import numpy as np
        from optparse import OptionParser
        import sys
        from time import time
        import matplotlib.pyplot as plt

        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.feature_selection import SelectFromModel
        from sklearn.feature_selection import SelectKBest, chi2
        from sklearn.linear_model import RidgeClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import Perceptron
        from sklearn.linear_model import PassiveAggressiveClassifier
        from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neighbors import NearestCentroid
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils.extmath import density
        from sklearn import metrics


        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')


        # parse commandline arguments
        op = OptionParser()
        op.add_option("--report",
                    action="store_true", dest="print_report",
                    help="Print a detailed classification report.")
        op.add_option("--chi2_select",
                    action="store", type="int", dest="select_chi2",
                    help="Select some number of features using a chi-squared test")
        op.add_option("--confusion_matrix",
                    action="store_true", dest="print_cm",
                    help="Print the confusion matrix.")
        op.add_option("--top10",
                    action="store_true", dest="print_top10",
                    help="Print ten most discriminative terms per class"
                        " for every classifier.")
        op.add_option("--all_categories",
                    action="store_true", dest="all_categories",
                    help="Whether to use all categories or not.")
        op.add_option("--use_hashing",
                    action="store_true",
                    help="Use a hashing vectorizer.")
        op.add_option("--n_features",
                    action="store", type=int, default=2 ** 16,
                    help="n_features when using the hashing vectorizer.")
        op.add_option("--filtered",
                    action="store_true",
                    help="Remove newsgroup information that is easily overfit: "
                        "headers, signatures, and quoting.")


        def is_interactive():
            return not hasattr(sys.modules['__main__'], '__file__')


        # work-around for Jupyter notebook and IPython console
        argv = [] if is_interactive() else sys.argv[1:]
        (opts, args) = op.parse_args(argv)
        if len(args) > 0:
            op.error("this script takes no arguments.")
            sys.exit(1)

        print(__doc__)
        op.print_help()
        print()


        # #############################################################################
        # Load some categories from the training set
        if opts.all_categories:
            categories = None
        else:
            categories = [
                'alt.atheism',
                'talk.religion.misc',
                'comp.graphics',
                'sci.space',
            ]

        if opts.filtered:
            remove = ('headers', 'footers', 'quotes')
        else:
            remove = ()

        print("Loading 20 newsgroups dataset for categories:")
        print(categories if categories else "all")

        data_train = fetch_20newsgroups(subset='train', categories=categories,
                                        shuffle=True, random_state=42,
                                        remove=remove)

        data_test = fetch_20newsgroups(subset='test', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=remove)
        print('data loaded')

        # order of labels in `target_names` can be different from `categories`
        target_names = data_train.target_names


        def size_mb(docs):
            return sum(len(s.encode('utf-8')) for s in docs) / 1e6


        data_train_size_mb = size_mb(data_train.data)
        data_test_size_mb = size_mb(data_test.data)

        print("%d documents - %0.3fMB (training set)" % (
            len(data_train.data), data_train_size_mb))
        print("%d documents - %0.3fMB (test set)" % (
            len(data_test.data), data_test_size_mb))
        print("%d categories" % len(target_names))
        print()

        # split a training set and a test set
        y_train, y_test = data_train.target, data_test.target

        print("Extracting features from the training data using a sparse vectorizer")
        t0 = time()
        if opts.use_hashing:
            vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                        n_features=opts.n_features)
            X_train = vectorizer.transform(data_train.data)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                        stop_words='english')
            X_train = vectorizer.fit_transform(data_train.data)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
        print("n_samples: %d, n_features: %d" % X_train.shape)
        print()

        print("Extracting features from the test data using the same vectorizer")
        t0 = time()
        X_test = vectorizer.transform(data_test.data)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
        print("n_samples: %d, n_features: %d" % X_test.shape)
        print()

        # mapping from integer feature name to original token string
        if opts.use_hashing:
            feature_names = None
        else:
            feature_names = vectorizer.get_feature_names()

        if opts.select_chi2:
            print("Extracting %d best features by a chi-squared test" %
                opts.select_chi2)
            t0 = time()
            ch2 = SelectKBest(chi2, k=opts.select_chi2)
            X_train = ch2.fit_transform(X_train, y_train)
            X_test = ch2.transform(X_test)
            if feature_names:
                # keep selected feature names
                feature_names = [feature_names[i] for i
                                in ch2.get_support(indices=True)]
            print("done in %fs" % (time() - t0))
            print()

        if feature_names:
            feature_names = np.asarray(feature_names)


        def trim(s):
            return s if len(s) <= 80 else s[:77] + "..."


        # #############################################################################
        # Benchmark classifiers
        def benchmark(clf):
            print('_' * 80)
            print("Training: ")
            print(clf)
            t0 = time()
            clf.fit(X_train, y_train)
            train_time = time() - t0
            print("train time: %0.3fs" % train_time)

            t0 = time()
            pred = clf.predict(X_test)
            test_time = time() - t0
            print("test time:  %0.3fs" % test_time)

            score = metrics.accuracy_score(y_test, pred)
            print("accuracy:   %0.3f" % score)

            if hasattr(clf, 'coef_'):
                print("dimensionality: %d" % clf.coef_.shape[1])
                print("density: %f" % density(clf.coef_))

                if opts.print_top10 and feature_names is not None:
                    print("top 10 keywords per class:")
                    for i, label in enumerate(target_names):
                        top10 = np.argsort(clf.coef_[i])[-10:]
                        print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
                print()

            if opts.print_report:
                print("classification report:")
                print(metrics.classification_report(y_test, pred,
                                                    target_names=target_names))

            if opts.print_cm:
                print("confusion matrix:")
                print(metrics.confusion_matrix(y_test, pred))

            print()
            clf_descr = str(clf).split('(')[0]
            return clf_descr, score, train_time, test_time


        results = []
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
                (PassiveAggressiveClassifier(max_iter=50, tol=1e-3),
                "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(n_estimators=100), "Random forest")):
            print('=' * 80)
            print(name)
            results.append(benchmark(clf))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                            tol=1e-3)))

            # Train SGD model
            results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                                penalty=penalty)))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                            penalty="elasticnet")))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(benchmark(MultinomialNB(alpha=.01)))
        results.append(benchmark(BernoulliNB(alpha=.01)))
        results.append(benchmark(ComplementNB(alpha=.1)))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(benchmark(Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                        tol=1e-3))),
        ('classification', LinearSVC(penalty="l2"))])))

        # make some plots

        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]

        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        plt.barh(indices + .3, training_time, .2, label="training time",
                color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()
    """)
    selenium.run(cmd)