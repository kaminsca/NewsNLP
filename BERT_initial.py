from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def preprocess(data):
    #TODO: change from just using title eventually
    text = data["title"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # using https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=nJ3Teyjmank2
    # labels_batch = {k: data[k] for k in data.keys() if k in labels}
    # # create numpy array of shape (batch_size, num_labels)
    # labels_matrix = np.zeros((len(text), len(labels)))
    # # fill numpy array
    # for idx, label in enumerate(labels):
    #     labels_matrix[:, idx] = labels_batch[label]
    #
    # encoding["labels"] = labels_matrix.tolist()
    return encoding

if __name__ == "__main__":
    # load data into train and test datasets
    dataset = load_dataset("csv", data_files='./output/master_data_no_article_content.csv')
    train_test_split_dataset = dataset['train'].train_test_split(test_size=0.2)
    train_dataset = train_test_split_dataset['train']
    test_dataset = train_test_split_dataset['test']
    print("Train dataset head:")
    print(train_dataset[0])

    # create labels and maps from index to label and vice versa
    labels = np.unique(train_dataset["county"])
    print(labels)
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    encoded_dataset = train_dataset.map(preprocess, batched=True)
