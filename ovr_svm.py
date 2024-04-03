from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
from joblib import dump

if __name__ == "__main__":
    #https://www.capitalone.com/tech/machine-learning/scikit-tfidf-implementation/
    df = pd.read_csv('./output/processed_data.csv', delimiter='|')

    corpus = df['content'].fillna('').to_list()
    print(corpus[:10])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names_out()) 
    Y = df.iloc[:, 5:].values
    print("done")

    clf = OneVsRestClassifier(SVC(kernel='sigmoid', verbose=True)).fit(X,Y)
    dump(vectorizer, 'tfidf.joblib')
    dump(clf, 'ovr-svm.joblib')
    # test_text = ["Trump goes to Zoo"]
    # test_vector = vectorizer.transform(test_text)
    # prediction = clf.predict(test_vector)
    # print(f"prediction: {prediction}")