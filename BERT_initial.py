from transformers import AutoTokenizer
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def preprocess(data):
    text = data[2]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    return encoding

if __name__ == "__main__":
    # dataset = load_dataset('./output/master_data_no_article_content.csv')
    dataset = load_dataset("csv", data_files='./output/master_data_no_article_content.csv')
    dataset
    # X = []
    # y = []
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    #
    # encoded_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset['train'].column_names)
    # preprocess(dataset)