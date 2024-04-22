from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,recall_score,f1_score, confusion_matrix
from sklearn.svm import SVC
import pandas as pd
from joblib import dump, load
import numpy as np
import os

prefix = os.path.dirname(os.path.abspath(__file__))

def analyze_hyperparams(X, Y):
    res = []
    #https://scikit-learn.org/stable/modules/cross_validation.html
    #https://medium.com/@24littledino/support-vector-machine-svm-in-python-fc3a4ffd25b6
    for kernel in ['linear','poly','rbf','sigmoid']:
        for C in [0.5,1.0,1.5,2.0,2.5,3.0]:
            svc = SVC(kernel=kernel, C=C)
            #clf = OneVsRestClassifier(svc)
            score = cross_val_score(svc, X,Y,cv=5)
            res.append((kernel, C, score.mean()))
    return res

def produce_svm_model(X, Y, kernel='linear', C=1.5, verbose=True, vectorizer_name='./output/tfidf.joblib', clf_name='./output/ovr-svm.joblib'):
    clf = OneVsRestClassifier(SVC(kernel=kernel, verbose=verbose, C=C)).fit(X, Y)
    dump(vectorizer, vectorizer_name)
    dump(clf, clf_name)

def svc_binary_clf(X,Y, kernel='linear', C=1.5, verbose=True, vectorizer_name='tfidf_single.joblib', clf_name='svm.joblib', vectorizer=None):
    svc = SVC(kernel=kernel, verbose=verbose, C=C).fit(X,Y)
    dump(vectorizer, vectorizer_name)
    dump(svc, clf_name)

def calc_svm_metrics(x_test, y_test, aggregator, vectorizer_name='tfidf_single.joblib', clf_name='ovr-svm.joblib'):
    clf = load(clf_name)
    vectorizer = load(vectorizer_name)
    y_pred = clf.predict(x_test)
    #https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    cm = confusion_matrix(y_test, y_pred)

    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    aggregator['fp'] += fp
    aggregator['fn'] += fn
    aggregator['tp'] += tp
    aggregator['tn'] += tn

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    micro_precision = tp / (tp + fp)
    micro_recall = tp / (tp + fn)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    print(f"micro averaged metrics for: {clf_name}")
    print(f"accuracy - {accuracy}")
    print(f"precision - {micro_precision}")
    print(f"recall - {micro_recall}")
    print(f"f1 - {micro_f1}")



if __name__ == "__main__":
    #https://www.capitalone.com/tech/machine-learning/scikit-tfidf-implementation/
    #train_df = pd.read_csv(prefix + '/data/train.csv', delimiter='|', nrows=20000)
    #test_df = pd.read_csv(prefix + '/data/test.csv', delimiter='|', nrows=20000)
    df = pd.read_csv('./output/val.csv', delimiter='|', nrows=2000)

    corpus = df['content'].fillna('').to_list()
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    for i in range(3):
        Y = df.iloc[:,5+i].values
        res = analyze_hyperparams(X, Y)
        print(f"Results for SVM -{i}")
        for kernel, C, score in res:
            print(f"{kernel}:{C}:{score}")
    # print('vectorized')
    # x_train = X[:20000]
    # x_test = X[20000:]

    # aggregator = {'tp': np.zeros(2), 'fp': np.zeros(2), 'fn': np.zeros(2), 'tn': np.zeros(2)}
    # #go through svm classifiers
    # for i in range(3):
    #     y_train = train_df.iloc[:, 5+i].values
    #     y_test = test_df.iloc[:, 5+i].values
    #     #TODO: update to fit data (Josh)
    #     # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=65)
    #     #train each classifier
    #     svc_binary_clf(x_train,y_train, clf_name=f"svm-{i}.joblib", vectorizer=vectorizer)
    #     #compute the metrics
    #     #TODO: Needs to use validation set instead
    #     calc_svm_metrics(x_test, y_test, aggregator, clf_name=f"svm-{i}.joblib")

    # TP = aggregator['tp'].sum()
    # FP = aggregator['fp'].sum()
    # FN = aggregator['fn'].sum()
    # TN = aggregator['tn'].sum()


    # print(f"{TP} - {FP} - {FN} - {TN}")
    # accuracy = (TP+TN)/(TP+TN+FP+FN)
    # micro_precision = TP / (TP + FP)
    # micro_recall = TP / (TP + FN)
    # micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    # print("metrics(average=micro)")
    # print(f"accuracy-{accuracy}")
    # print(f"precision-{micro_precision}")
    # print(f"recall - {micro_recall}")
    # print(f"F1-score - {micro_f1}")