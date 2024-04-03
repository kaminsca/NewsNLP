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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

if __name__ == "__main__":
    # https://medium.com/@abdurhmanfayad_73788/fine-tuning-bert-for-a-multi-label-classification-problem-on-colab-5ca5b8759f3f
    df = pd.read_csv('./output/processed_data.csv', delimiter='|')
    train_df, test_df = train_test_split(df, test_size=0.2)
    print("Train dataset head:")
    print(train_df)


    columns = ["avg_white_pop_pct","avg_median_hh_inc","avg_non_college_pct"]
    df_labels_train = train_df[columns]
    df_labels_test = test_df[columns]

    #convert to label lists
    labels_list_train = df_labels_train.values.tolist()
    labels_list_test = df_labels_test.values.tolist()

    # set up our text inputs
    train_texts = train_df['content'].tolist()
    train_labels = labels_list_train

    eval_texts = test_df['content'].tolist()
    eval_labels = labels_list_test

    # print(train_labels)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    #TODO: increase max length when we use article content
    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=64)
    eval_encodings = tokenizer(eval_texts, padding="max_length", truncation=True, max_length=64)

    #print(train_encodings[0])

    train_dataset = TextClassifierDataset(train_encodings, train_labels)
    eval_dataset = TextClassifierDataset(eval_encodings, eval_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=3
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

    ## Evaluate the model
    results = trainer.evaluate()
    print(results)