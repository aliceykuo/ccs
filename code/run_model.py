from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import MultinomialNB

'''
DOC:
This file contains all the models that were tested to determine most effective classification method.

'''


def pca(X):
    pca = RandomizedPCA(n_components=250)
    X = pca.fit_transform(X)

def random_forest(X_train, X_test, y_train, y_test):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)

    print "rf precision:", precision_score(y_test, y_predict)
    print "rf recall:", recall_score(y_test, y_predict)
    print "rf f1:", f1_score(y_test, y_predict, average='weighted')
    return rf

def logistic_regression(X_train, X_test, y_train, y_test):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lr = LogisticRegression(C=100)
    lr.fit(X_train, y_train)
    probabilities = lr.predict_proba(X_test)[:]
    y_predict = lr.predict(X_test)
    print probabilities
    print "lr precision:", precision_score(y_test, y_predict)
    print "lr recall:", recall_score(y_test, y_predict)
    print "lr f1:", f1_score(y_test, y_predict, average='weighted')
    return lr


def svm(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    # probabilities = svm.predict_proba(X_test)[:]
    y_predict = svm.predict(X_test)

    print "svm precision:", precision_score(y_test, y_predict)
    print "svm recall:", recall_score(y_test, y_predict)
    print "svm f1:", f1_score(y_test, y_predict, average='weighted')

def knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)

    print "knn precision:", precision_score(y_test, y_predict)
    print "knn recall:", recall_score(y_test, y_predict)
    print "knn f1:", f1_score(y_test, y_predict, average='weighted')

def gradient_descent(X_train, X_test, y_train, y_test):
    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    # probabilities = sgd.predict_proba(X_test)[:]
    y_predict = sgd.predict(X_test)

    print "sgd precision:", precision_score(y_test, y_predict)
    print "sgd recall:", recall_score(y_test, y_predict)
    print "sgd f1:", f1_score(y_test, y_predict, average='weighted')

def onevsrest(X_train, X_test, y_train, y_test):
    onevsrest = OneVsRestClassifier(LinearSVC(random_state=0))
    onevsrest.fit(X_train, y_train)
    y_predict = onevsrest.predict(X_test)

    print "onevsrest precision:", precision_score(y_test, y_predict)
    print "onevsrest recall:", recall_score(y_test, y_predict)
    print "onevsrest f1:", f1_score(y_test, y_predict, average='weighted')

def rbm(X_train, X_test, y_train, y_test):
    rbm = BernoulliRBM(n_components=3000)
    rbm.fit(X_train, y_train)


def grid_search(est, grid):
    rbm = GridSearchCV(est, grid, n_jobs=-1, verbose=True,
                          scoring='mean_squared_error').fit(X_train, y_train)
    return grid_cv

def multinomial_naivebayes(X_train, X_test, y_train, y_test):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print "f1:", f1_score(y_test, y_predict, average='weighted')
    print f1_scores
    print sum(f1_scores) / len(f1_scores)

if __name__ == '__main__':
    pkl_path1 = '../extract_patches25_500img_scaled_0350.pkl'
    df = pd.read_pickle(pkl_path1)
    X = df[:, 1:]
    y = df[:, 0:1]

    print X.shape
    print y.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf = random_forest(X_train, X_test, y_train, y_test)
    lr = logistic_regression(X_train, X_test, y_train, y_test)
    svm = svm(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)
    gradient_descent(X_train, X_test, y_train, y_test)
    onevsrest(X_train, X_test, y_train, y_test)

    grid = {'max_depth': [5, None],
                'max_features': [100, 500],
                'n_estimators': [10, 20],
                'random_state': [1]}

    grid_search = grid_search(lr, grid)
    print grid_search
    best_est = grid_search.best_estimator_

    gs_svm = GridSearchCV(svm,{'C':np.linspace(.1, 2, 10),'degree':range(2,7)},scoring='accuracy',cv=10)
    print "grid search SVM", gs_svm
    print "svm best params", svm.best_params_
    print "grid scores", svm.grid_scores_
    score = []
    for i in xrange(50):
        svm =  svm(X_train, X_test, y_train, y_test)

    pkl.dump(rf, open('model/baseline.pkl', 'wb'))


