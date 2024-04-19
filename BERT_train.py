from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import torch
import wandb


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

def evaluate(pred):
        preds, labels = pred
        # get loss from preds and labels
        labels = torch.as_tensor(labels)
        preds = (torch.sigmoid(torch.as_tensor(preds)) >= 0.5).int()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds.float(), labels, reduction='none')
        f1 = metrics.f1_score(labels.view(-1), preds.view(-1))
        prec = metrics.precision_score(labels.view(-1), preds.view(-1))
        rec = metrics.recall_score(labels.view(-1), preds.view(-1))
        mean_loss = torch.mean(loss).item()
        
        log = {"avg_loss": mean_loss, "f1": f1, "precision": prec, "recall": rec}
        # if flag: wandb.log(log)
        return log


if __name__ == "__main__":
    # https://medium.com/@abdurhmanfayad_73788/fine-tuning-bert-for-a-multi-label-classification-problem-on-colab-5ca5b8759f3f
    train_df = pd.read_csv('./output/train.csv', delimiter='|')
    test_df = pd.read_csv('./output/test.csv', delimiter='|')
    print("Train dataset head:")
    print(train_df.head())

    columns = ["avg_white_pop_pct","avg_median_hh_inc","avg_non_college_pct"]
    df_labels_train = train_df[columns]
    df_labels_test = test_df[columns]

    #convert to label lists
    train_labels = df_labels_train.values.tolist()
    test_labels = df_labels_test.values.tolist()

    # set up our text inputs
    train_texts = train_df['content'].tolist()

    test_texts = test_df['content'].tolist()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    #TODO: increase max length when we use article content
    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=64)
    test_encodings = tokenizer(test_texts, padding="max_length", truncation=True, max_length=64)

    train_dataset = TextClassifierDataset(train_encodings, train_labels)
    test_dataset = TextClassifierDataset(test_encodings, test_labels)
    
    print("Datasets created")
        
    # flag = True
    # # wandb.init(
    # #     # set the wandb project where this run will be logged
    # #     project="news-nlp",
        
    # #     # track hyperparameters and run metadata
    # #     config={
    # #         "epochs": 4,
    # #     }
    # # )
    wandb.init(
        # set the wandb project where this run will be logged
        project="news-nlp",
        
        # track hyperparameters and run metadata
        config={
            "epochs": 4,
        }
    )
    flag=True
    steps=0
    
    def evaluate(pred):
        preds, labels = pred
        # get loss from preds and labels
        labels = torch.as_tensor(labels)
        preds = (torch.sigmoid(torch.as_tensor(preds)) >= 0.5).int()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds.float(), labels, reduction='none')
        f1 = metrics.f1_score(labels.view(-1), preds.view(-1))
        prec = metrics.precision_score(labels.view(-1), preds.view(-1))
        rec = metrics.recall_score(labels.view(-1), preds.view(-1))
        mean_loss = torch.mean(loss).item()
        steps += 1000
        
        log = {"avg_loss": mean_loss, "f1": f1, "precision": prec, "recall": rec, "steps": steps}
        if flag: wandb.log(log)
        return log

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "bert-base-uncased",
    #     problem_type="multi_label_classification",
    #     num_labels=3
    # )

    training_arguments = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="steps",
        eval_steps=1000,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        save_strategy='epoch',
        save_only_model=True
    )

    model = AutoModelForSequenceClassification.from_pretrained('./checkpoint_4500')

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=evaluate
    )
    
    print("Training")

    # trainer.train()
    # wandb.finish()
    flag = False
    
    print("Saving")
    
    trainer.save_model(output_dir='./trained_bert')

    ## Evaluate the model
    results = trainer.evaluate()
    print(results)
    
    trainer.eval_dataset = test_dataset[:3]
    res = trainer.predict()
    print(res)
    print(trainer.eval_dataset)
    