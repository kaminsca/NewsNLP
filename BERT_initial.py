from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
import torch


class TextClassifierDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# def preprocess(data, tokenizer, labels):
#     #TODO: change from just using title eventually
#     text = data["title"]
#     encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
#     # using https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=nJ3Teyjmank2
#     labels_batch = {k: data[k] for k in data.keys() if k in labels}
#     # create numpy array of shape (batch_size, num_labels)
#     labels_matrix = np.zeros((len(text), len(labels)))
#     # fill numpy array
#     for idx, label in enumerate(labels):
#         labels_matrix[:, idx] = labels_batch[label]
    
#     encoding["labels"] = labels_matrix.tolist()
#     return encoding


if __name__ == "__main__":
    # load data into train and test datasets
    # dataset = load_dataset("csv", data_files='./output/master_data_no_content.csv', delimiter='|')
    # example = dataset['train'][0]
    # print(example)
    # print(dataset)
    # train_test_split_dataset = dataset['train'].train_test_split(test_size=0.2)
    # train_dataset = train_test_split_dataset['train']
    # test_dataset = train_test_split_dataset['test']
    # print("Train dataset head:")
    # print(train_dataset[0])

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # encoded_dataset = train_dataset.map(lambda examples: preprocess(examples, tokenizer, labels), batched=True)
    #encoded_dataset = train_dataset.map(preprocess, batched=True)

    # https://medium.com/@abdurhmanfayad_73788/fine-tuning-bert-for-a-multi-label-classification-problem-on-colab-5ca5b8759f3f
    df = pd.read_csv('./output/master_data_no_content.csv', delimiter='|')
    train_df, test_df = train_test_split(df, test_size=0.2)
    print("Train dataset head:")
    print(train_df)

    # set up labels
    # create map from index to label and vice versa
    labels = np.unique(train_df["county"])
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    train_df['label_id'] = train_df['county'].map(label2id)
    test_df['label_id'] = test_df['county'].map(label2id)

    labels_train = train_df['label_id']
    labels_list_train = labels_train.values.tolist()
    labels_test = test_df['label_id']
    labels_list_test = labels_test.values.tolist()
    # print(labels_list_train)

    # set up our text inputs
    train_texts = train_df['title'].tolist()
    train_labels = labels_list_train

    eval_texts = test_df['title'].tolist()
    eval_labels = labels_list_test
    # print(train_labels)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    #TODO: increase max length when we use article content
    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=64)
    eval_encodings = tokenizer(eval_texts, padding="max_length", truncation=True, max_length=64)

    print(train_encodings[0])

    train_dataset = TextClassifierDataset(train_encodings, train_labels)
    eval_dataset = TextClassifierDataset(eval_encodings, eval_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="single_label_classification",
        num_labels=len(labels_list_train)
    )

    training_arguments = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4
    )

    trainer = Trainer(
        model = model,
        args = training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model(output_dir='./trained_bert')

    # Evaluate the model
    results = trainer.evaluate()
    print(results)
    
