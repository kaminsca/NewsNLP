import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt

def plot_pca():
    #read in data to be vectorized
    df = pd.read_csv('./output/processed_data.csv', delimiter='|')
    corpus = df['content'].fillna('').to_list()
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)

    svd = TruncatedSVD(n_components=2)
    df2d = pd.DataFrame(svd.fit_transform(X), columns=list('xy'))

    df2d.plot(kind='scatter', x='x', y='y')

    plt.show()

def combine_label(row):
    ret_arr = []
    for i in range(5,8):
        ret_arr.append(str(row[i]))
    return '_'.join(ret_arr)

def compute_label_counts(df):
    df['combined_multi_label'] = df.apply(combine_label, axis=1)
    label_combo_counts = df['combined_multi_label'].value_counts()
    print(label_combo_counts)

def compute_individual_labels(df):
    label_count = []
    for i in range(5, 8):
        label_count.append(df.iloc[:,i].value_counts())
    
    for idx, itm in enumerate(label_count):
        print(f"label: {idx} | count: {itm}")

if __name__ == "__main__":
    df = pd.read_csv('./output/processed_data.csv', delimiter='|')
    #compute_label_counts(df)
    compute_individual_labels(df)
