import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, TFAutoModel
import os
from tqdm import tqdm

prefix = os.path.dirname(os.path.abspath(__file__))

train_df = pd.read_csv(prefix + '/data/train.csv', delimiter='|')
val_df = pd.read_csv(prefix + '/data/val.csv', delimiter='|')
test_df = pd.read_csv(prefix + '/data/test.csv', delimiter='|')
all_df = pd.concat([train_df, val_df, test_df])
print(len(all_df))

# using tfidf for now
vectorizer = TfidfVectorizer(stop_words='english')
all_tfidf = vectorizer.fit_transform(all_df['content'])
X_train = all_tfidf[:100000]
X_val = all_tfidf[100000:135000]
X_test = all_tfidf[135000:]
print('vectorized')

labels = ['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']
n_neighbors = [15]
yts = []
yps = []
accs = []
precs = []
f1s = []
for i, label in enumerate(labels):
    
    yts.append([])
    yps.append([])
    accs.append([])
    precs.append([])
    f1s.append([])
    
    for neighbors in tqdm(n_neighbors, desc=f"on label {label}"):
        # train
        knn = KNeighborsClassifier(n_neighbors=neighbors, metric='cosine')
        y_train = train_df[label]
        knn.fit(X_train, y_train)
        
        # evaluate
        y_pred = knn.predict(X_val)
        y_true = val_df[label]
        yts[i].append(y_true)
        yps[i].append(y_pred)
        
        acc = accuracy_score(y_true, y_pred)*100
        prec = precision_score(y_true, y_pred)*100
        f1 = f1_score(y_true, y_pred)*100
        accs[i].append(acc)
        precs[i].append(prec)
        f1s[i].append(f1)
        
        print(f'({label}) With k={neighbors}, Accuracy: {acc}, Precision: {prec}, F1: {f1}')

multi_yts = list(zip(yts[0][0], yts[1][0], yts[2][0]))
multi_yps = list(zip(yps[0][0], yps[1][0], yps[2][0]))
micro_f1 = f1_score(multi_yts, multi_yps, average='micro')

print(accs)
print(precs)
print(f1s)
print(micro_f1)

# labels = ['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']
# n_neighbors = [1, 3, 5, 7, 15, 30]
# yts = []
# yps = []
# accs = []
# f1s = []
# for i, label in enumerate(labels):
    
#     yts.append([])
#     yps.append([])
#     accs.append([])
#     f1s.append([])
    
#     for neighbors in tqdm(n_neighbors, desc=f"on label {label}"):
#         # train
#         knn = KNeighborsClassifier(n_neighbors=neighbors, metric='cosine')
#         y_train = train_df[label]
#         knn.fit(X_train, y_train)
        
#         # evaluate
#         y_pred = knn.predict(X_val)
#         y_true = val_df[label]
#         yts[i].append(y_true)
#         yps[i].append(y_pred)
        
#         acc = accuracy_score(y_true, y_pred)*100
#         f1 = f1_score(y_true, y_pred)*100
#         accs[i].append(acc)
#         f1s[i].append(f1)
        
#         print(f'({label}) With k={neighbors}, Accuracy: {acc}, F1: {f1}')

# multi_yts = [list(zip(yts[0][i], yts[1][i], yts[2][i])) for i in range(len(n_neighbors))]
# multi_yps = [list(zip(yps[0][i], yps[1][i], yps[2][i])) for i in range(len(n_neighbors))]
# micro_f1s = [f1_score(multi_yts[i], multi_yps[i], average='micro') for i in range(len(multi_yts))]

# topn_acc = list(set([acc.index(max(acc)) for acc in accs]))
# topn_f1 = list(set([f1.index(max(f1)) for f1 in f1s]))
# topn_f1m = micro_f1s.index(max(micro_f1s))
# print(topn_acc)
# print(topn_f1)
# print(topn_f1m)








# accs = []
# f1s = []

# for i, label in enumerate(labels):
    
#     accs.append([])
#     for neighbors in tqdm(top_n, desc=f"on label {label}"):
#         # train
#         knn = KNeighborsClassifier(n_neighbors=neighbors, metric='cosine')
#         knn.fit(X_train, y_train)
#         # evaluate
#         y_pred = knn.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)*100
#         accs[i].append(acc)
#         print(f'({label}) Accuracy with k={neighbors}', acc)



# train_df = pd.read_csv('./output/train.csv', delimiter='|')
# test_df = pd.read_csv('./output/test.csv', delimiter='|')
# print('read')
# train_df['combined_label'] = train_df[['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
# test_df['combined_label'] = train_df[['avg_white_pop_pct', 'avg_median_hh_inc', 'avg_non_college_pct']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
# # print(df.head())
# # using tfidf for now
# vectorizer = TfidfVectorizer(stop_words='english')
# print('vectorizing')
# X_train = vectorizer.fit_transform(train_df['content'])
# # X_test = vectorizer.fit_transform(test_df['content'])
# # y_train = train_df['combined_label']
# # y_test = test_df['combined_label']
# # print(X_train.shape)
# # print(X_test.shape)

# # print('vectorized')


# # # lets see if bert encodings are better
# # # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# # # model = TFAutoModel.from_pretrained("bert-base-uncased")
# # # encodings = tokenizer(df['content'].tolist(), padding="max_length", truncation=True, max_length=64, return_tensors="tf")
# # # # using the `[CLS]` token's embedding
# # # bert_embeddings= model(encodings).last_hidden_state[:, 0, :].numpy() 

# # # train
# # knn1 = KNeighborsClassifier(n_neighbors=1, metric='cosine')
# # knn5 = KNeighborsClassifier(n_neighbors=5, metric='cosine')
# # knn3 = KNeighborsClassifier(n_neighbors=3, metric='cosine')
# # knn7 = KNeighborsClassifier(n_neighbors=7, metric='cosine')
# # knn15 = KNeighborsClassifier(n_neighbors=15, metric='cosine')
# # knn1.fit(X_train, y_train)
# # print('done knn1')
# # knn5.fit(X_train, y_train)
# # print('done knn5')
# # knn3.fit(X_train, y_train)
# # print('done knn3')
# # knn7.fit(X_train, y_train)
# # print('done knn7')
# # knn15.fit(X_train, y_train)
# # print('done knn15')

# # # evaluate
# # y_pred_1 = knn1.predict(X_test)
# # y_pred_5 = knn5.predict(X_test)
# # y_pred_3 = knn3.predict(X_test)
# # y_pred_7 = knn7.predict(X_test)
# # y_pred_15 = knn15.predict(X_test)
# # print("Accuracy with k=1", accuracy_score(y_test, y_pred_1)*100)
# # print("Accuracy with k=3", accuracy_score(y_test, y_pred_3)*100)
# # print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
# # print("Accuracy with k=7", accuracy_score(y_test, y_pred_7)*100)
# # print("Accuracy with k=15", accuracy_score(y_test, y_pred_15)*100)
