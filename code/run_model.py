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

def random_forest(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    mean_f1 = cross_val_score(rf, X, y, cv=5, scoring='f1').mean()

    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    rf.fit(X, y)
    pkl.dump(rf, open('/Users/kuoyen/Desktop/wedding/rf_model_500.pkl', 'wb'))
    # rf.predict

    # pkl.dump(rf, open('rf.pkl', 'wb'))
    print "RANDOM FOREST F1:", mean_f1
    return rf

# def predict(X):
    
#     labels = rf.predict(X)
#     print labels 
#     answers = np.array([Counter(row).most_common(1)[0][0] for row in labels])
#     return abswers


if __name__ == '__main__':
    # pkl_path1 = '/Users/kuoyen/Documents/capstone/images/pkl/mat_50_1000_minmaxscaler.pkl'
    pkl_path1 = '/Users/kuoyen/Desktop/wedding/extract_patches25_500img_0310.pkl'
    # pkl_path1 = '/Users/kuoyen/Desktop/wedding/extract_patches_1000img_0437.pkl'

    # finished pickling ../images/pkl/mat_30_1000_nonscaled.pkl
    # finished pickling ../images/pkl/mat_30_1000_stdscaled.pkl
    # pkl_path2 = '/Users/kuoyen/Documents/capstone/images/pkl/mat_50_1000_stdscaler.pkl'
    df = pd.read_pickle(pkl_path1)
    X = df[:, 1:]
    y = df[:, 0:1]

    print X.shape
    print y.shape
    rf = random_forest(X, y.ravel())


    # /Users/kuoyen/Desktop/wedding/extract_patches25_50img_1611.pkl