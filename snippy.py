# coding=utf-8
# scikit learn 0.17 test


digits.target
digits.data


def example1():
    from sklearn import datasets
    iris = datasets.load_iris()
    digits = datasets.load_digits()
    from sklearn import svm
    from sklearn import svm
    clf = svm.SVC(gamma=0.001, C=100.)
    # model build
    clf.fit(digits.data[:-1], digits.target[:-1])  # skip the last record
    # predict
    clf.predict(digits.data[-1:])  # predict the last record


def example2():
    """
    Model persistence
    """
    from sklearn import svm
    from sklearn import datasets
    clf = svm.SVC()
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    import pickle
    s = pickle.dumps(clf)
    clf2 = pickle.loads(s)
    clf2.predict(X[0:1])
    # more efficient
    from sklearn.externals import joblib
    joblib.dump(clf, 'filename.pkl')
    clf = joblib.load('filename.pkl')
