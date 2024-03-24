import pandas as pd
import csv
import sqlite3
from transformers import AutoModel, AutoTokenizer, AutoConfig


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model_path = './trained_bert'
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    text= 'Trump blasts 3M as company says mask demand far exceeds ability to produce the'
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

    outputs = trainer.model(**encoding)
    logits = outputs.logits
    
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(predicted_labels)