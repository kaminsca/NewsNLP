import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, TFAutoModel

df = pd.read_csv('./output/processed_data.csv', delimiter='|', nrows=2000)
df['combined_label'] = df[['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
# print(df.head())
# using tfidf for now
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(df['content'])
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['combined_label'], test_size=0.2, random_state=42)

# lets see if bert encodings are better
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
model = TFAutoModel.from_pretrained("bert-base-uncased")
encodings = tokenizer(df['content'].tolist(), padding="max_length", truncation=True, max_length=64, return_tensors="tf")
# using the `[CLS]` token's embedding
bert_embeddings= model(encodings).last_hidden_state[:, 0, :].numpy() 
X_train, X_test, y_train, y_test = train_test_split(bert_embeddings, df['combined_label'], test_size=0.2, random_state=42)

# train
knn1 = KNeighborsClassifier(n_neighbors=1, metric='cosine')
knn5 = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn3 = KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn7 = KNeighborsClassifier(n_neighbors=7, metric='cosine')
knn15 = KNeighborsClassifier(n_neighbors=15, metric='cosine')
knn1.fit(X_train, y_train)
knn5.fit(X_train, y_train)
knn3.fit(X_train, y_train)
knn7.fit(X_train, y_train)
knn15.fit(X_train, y_train)

# evaluate
y_pred_1 = knn1.predict(X_test)
y_pred_5 = knn5.predict(X_test)
y_pred_3 = knn3.predict(X_test)
y_pred_7 = knn7.predict(X_test)
y_pred_15 = knn15.predict(X_test)
print("Accuracy with k=1", accuracy_score(y_test, y_pred_1)*100)
print("Accuracy with k=3", accuracy_score(y_test, y_pred_3)*100)
print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
print("Accuracy with k=7", accuracy_score(y_test, y_pred_7)*100)
print("Accuracy with k=15", accuracy_score(y_test, y_pred_15)*100)
