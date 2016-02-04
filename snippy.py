# coding=utf-8
# scikit learn 0.17 test

def linearReg():
    from sklearn import datasets
    diabetes = datasets.load_diabetes()
    diabetes_X_train = diabetes.data[:-20]
    diabetes_X_test = diabetes.data[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    print(regr.coef_)
    import numpy as np
    np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)
    regr.score(diabetes_X_test, diabetes_y_test)

    X = np.c_[.5, 1].T
    y = [.5, 1]
    test = np.c_[0, 2].T
    regr = linear_model.LinearRegression()

    import pylab as pl
    pl.figure()
    np.random.seed(0)
    for _ in range(6):
        this_X = .1 * np.random.normal(size=(2, 1)) + X
        regr.fit(this_X, y)
        pl.plot(test, regr.predict(test))
        pl.scatter(this_X, y, s=3)


def autoRefittingParameters():
    """
    Reﬁtting and updating parameters
    :return:
    """
    import numpy as np
    from sklearn.svm import SVC
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    y = rng.binomial(1, 0.5, 100)
    X_test = rng.rand(5, 10)

    clf = SVC()
    clf.set_params(kernel='linear').fit(X, y)
    clf.predict(X_test)
    clf.set_params(kernel='rbf').fit(X, y)
    clf.predict(X_test)

def example1():
    from sklearn import datasets
    iris = datasets.load_iris()
    digits = datasets.load_digits()
    digits.target
    digits.data

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


def ex3():
    """
    scikit conventions
    :return:
    """
    import numpy as np
    from sklearn import random_projection

    rng = np.random.RandomState(0)
    X = rng.rand(10, 2000)
    X = np.array(X, dtype='float32')
    X.dtype
    # dtype('float32')

    # Type cast to float64
    transformer = random_projection.GaussianRandomProjection()
    X_new = transformer.fit_transform(X)
    X_new.dtype
    # dtype('float64')


def ex4():
    """
    type cast for fit function
    :return:
    """
    from sklearn import datasets
    from sklearn.svm import SVC
    iris = datasets.load_iris()
    clf = SVC()
    clf.fit(iris.data, iris.target)
    list(clf.predict(iris.data[:3]))
    clf.fit(iris.data, iris.target_names[iris.target])
    list(clf.predict(iris.data[:3]))
