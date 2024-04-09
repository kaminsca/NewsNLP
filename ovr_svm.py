from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.svm import SVC
import pandas as pd
from joblib import dump


def analyze_hyperparams(X, Y):
    res = []
    #https://scikit-learn.org/stable/modules/cross_validation.html
    #https://medium.com/@24littledino/support-vector-machine-svm-in-python-fc3a4ffd25b6
    for kernel in ['linear','poly','rbf','sigmoid']:
        for C in [0.5,1.0,1.5,2.0,2.5,3.0]:
            svc = SVC(kernel=kernel, verbose=True, C=C)
            clf = OneVsRestClassifier(svc)
            score = cross_val_score(clf, X,Y,cv=5)
            res.append((kernel, C, score.mean()))
    return res

def produce_svm_model(X, Y, kernel='linear', C=1.5, verbose=True, vectorizer_name='tfidf.joblib', clf_name='ovr-svm.joblib'):
    clf = OneVsRestClassifier(SVC(kernel=kernel, verbose=verbose, C=C)).fit(X, Y)
    dump(vectorizer, vectorizer_name)
    dump(clf, clf_name)

def display_svm_metrics(x_test, y_test, vectorizer_name='tfidf.joblib', clf_name='ovr-svm.joblib'):
    clf = load(clf_name)
    vectorizer = load(vectorizer_name)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred, average=None)
    f1_score = f1_score(y_test,y_pred, average=None)
    print("")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"f1: {f1_score}")

if __name__ == "__main__":
    #https://www.capitalone.com/tech/machine-learning/scikit-tfidf-implementation/
    df = pd.read_csv('./output/processed_data.csv', delimiter='|')

    corpus = df['content'].fillna('').to_list()


    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)

    Y = df.iloc[:, 5:].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=65)

    produce_svm_model(x_train,y_train)
    display_svm_metrics(x_test, y_test)