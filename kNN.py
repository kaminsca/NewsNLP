import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, TFAutoModel

# df = pd.read_csv('./output/processed_data.csv', delimiter='|')
# # df['combined_label'] = df[['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
# # using tfidf for now
# vectorizer = TfidfVectorizer(stop_words='english')
# X_tfidf = vectorizer.fit_transform(df['content'])
# # X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['combined_label'], test_size=0.2, random_state=42)
# # lets see if bert encodings are better
# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# # model = TFAutoModel.from_pretrained("bert-base-uncased")
# # encodings = tokenizer(df['content'].tolist(), padding="max_length", truncation=True, max_length=64, return_tensors="tf")
# # # using the `[CLS]` token's embedding
# # bert_embeddings= model(encodings).last_hidden_state[:, 0, :].numpy() 
# # print(bert_embeddings[0])
# # X_train, X_test, y_train, y_test = train_test_split(bert_embeddings, df['combined_label'], test_size=0.2, random_state=42)


# labels = ['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']
# for label in labels:
#     X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df[label], test_size=0.2, random_state=42)
#     n_neighbors = [1, 3, 5, 7, 15, 30]
#     for neighbors in n_neighbors:
#         # train
#         knn = KNeighborsClassifier(n_neighbors=neighbors, metric='cosine')
#         knn.fit(X_train, y_train)
#         # evaluate
#         y_pred = knn.predict(X_test)
#         print(f'({label}) Accuracy with k={neighbors}', accuracy_score(y_test, y_pred)*100)


train_df = pd.read_csv('./output/train.csv', delimiter='|')
test_df = pd.read_csv('./output/test.csv', delimiter='|')
print('read')
train_df['combined_label'] = train_df[['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
test_df['combined_label'] = train_df[['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
# print(df.head())
# using tfidf for now
vectorizer = TfidfVectorizer(stop_words='english')
print('vectorizing')
X_train = vectorizer.fit_transform(train_df['content'])
# X_test = vectorizer.fit_transform(test_df['content'])
# y_train = train_df['combined_label']
# y_test = test_df['combined_label']
# print(X_train.shape)
# print(X_test.shape)

# print('vectorized')
# # X_train, X_test, y_train, y_test = train_test_split(X_tfidf, train_df['combined_label'], test_size=0.1, random_state=42)


# # lets see if bert encodings are better
# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# # model = TFAutoModel.from_pretrained("bert-base-uncased")
# # encodings = tokenizer(df['content'].tolist(), padding="max_length", truncation=True, max_length=64, return_tensors="tf")
# # # using the `[CLS]` token's embedding
# # bert_embeddings= model(encodings).last_hidden_state[:, 0, :].numpy() 
# # X_train, X_test, y_train, y_test = train_test_split(bert_embeddings, df['combined_label'], test_size=0.2, random_state=42)

# # train
# knn1 = KNeighborsClassifier(n_neighbors=1, metric='cosine')
# knn5 = KNeighborsClassifier(n_neighbors=5, metric='cosine')
# knn3 = KNeighborsClassifier(n_neighbors=3, metric='cosine')
# knn7 = KNeighborsClassifier(n_neighbors=7, metric='cosine')
# knn15 = KNeighborsClassifier(n_neighbors=15, metric='cosine')
# knn1.fit(X_train, y_train)
# print('done knn1')
# knn5.fit(X_train, y_train)
# print('done knn5')
# knn3.fit(X_train, y_train)
# print('done knn3')
# knn7.fit(X_train, y_train)
# print('done knn7')
# knn15.fit(X_train, y_train)
# print('done knn15')

# # evaluate
# y_pred_1 = knn1.predict(X_test)
# y_pred_5 = knn5.predict(X_test)
# y_pred_3 = knn3.predict(X_test)
# y_pred_7 = knn7.predict(X_test)
# y_pred_15 = knn15.predict(X_test)
# print("Accuracy with k=1", accuracy_score(y_test, y_pred_1)*100)
# print("Accuracy with k=3", accuracy_score(y_test, y_pred_3)*100)
# print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
# print("Accuracy with k=7", accuracy_score(y_test, y_pred_7)*100)
# print("Accuracy with k=15", accuracy_score(y_test, y_pred_15)*100)
