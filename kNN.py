import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, TFAutoModel

df = pd.read_csv('./output/processed_data.csv', delimiter='|')
# df['combined_label'] = df[['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
# using tfidf for now
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(df['content'])
# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['combined_label'], test_size=0.2, random_state=42)
# lets see if bert encodings are better
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# model = TFAutoModel.from_pretrained("bert-base-uncased")
# encodings = tokenizer(df['content'].tolist(), padding="max_length", truncation=True, max_length=64, return_tensors="tf")
# # using the `[CLS]` token's embedding
# bert_embeddings= model(encodings).last_hidden_state[:, 0, :].numpy() 
# print(bert_embeddings[0])
# X_train, X_test, y_train, y_test = train_test_split(bert_embeddings, df['combined_label'], test_size=0.2, random_state=42)


labels = ['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']
for label in labels:
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df[label], test_size=0.2, random_state=42)
    n_neighbors = [1, 3, 5, 7, 15, 30]
    for neighbors in n_neighbors:
        # train
        knn = KNeighborsClassifier(n_neighbors=neighbors, metric='cosine')
        knn.fit(X_train, y_train)
        # evaluate
        y_pred = knn.predict(X_test)
        print(f'({label}) Accuracy with k={neighbors}', accuracy_score(y_test, y_pred)*100)