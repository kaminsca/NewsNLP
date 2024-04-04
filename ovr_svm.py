from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.svm import SVC
import pandas as pd
from joblib import dump

if __name__ == "__main__":
    #https://www.capitalone.com/tech/machine-learning/scikit-tfidf-implementation/
    df = pd.read_csv('./output/processed_data.csv', delimiter='|')

    corpus = df['content'].fillna('').to_list()



    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)

    Y = df.iloc[:, 5:].values

    x_train,x_test,y_train,y_test = train_test_split(X, df.iloc[:, 5:].values, test_size=0.3)


    clf = OneVsRestClassifier(SVC(kernel='rbf', verbose=True)).fit(x_train,y_train)
    dump(vectorizer, 'tfidf.joblib')
    dump(clf, 'ovr-svm.joblib')


    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred, average="macro")
    f1_score = f1_score(y_test,y_pred, average="macro")
    print("")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"f1: {f1_score}")