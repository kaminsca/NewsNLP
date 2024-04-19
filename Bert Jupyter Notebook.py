#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import wandb
from tqdm import tqdm


# In[3]:


# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# In[4]:


print(device)


# <a id='section02'></a>
# ### Importing and Pre-Processing the domain data
# 
# We will be working with the data and preparing for fine tuning purposes.
# *Assuming that the `train.csv` is already downloaded, unzipped and saved in your `data` folder*
# 
# * Import the file in a dataframe and give it the headers as per the documentation.
# * Taking the values of all the categories and coverting it into a list.
# * The list is appened as a new column and other columns are removed

# In[7]:


train_df = pd.read_csv("./data/train.csv", delimiter='|')
val_df = pd.read_csv("./data/val.csv", delimiter='|')
test_df = pd.read_csv("./data/test.csv", delimiter='|')
train_df['list'] = train_df[train_df.columns[-3:]].values.tolist()
val_df['list'] = val_df[val_df.columns[-3:]].values.tolist()
test_df['list'] = test_df[test_df.columns[-3:]].values.tolist()
train_dataset = train_df[['content', 'list']].copy()
val_dataset = val_df[['content', 'list']].copy()
test_dataset = test_df[['content', 'list']].copy()
train_dataset.head()


# In[8]:


# Sections of config

# Defining some key variables that will be used later on in the training
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-05
wandb_flag = False
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[9]:


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.content = self.data.content
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        content = str(self.content[index])
        content = " ".join(content.split())

        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']


        return {
            'ids': torch.tensor(ids, dtype=torch.long).to(device),
            'mask': torch.tensor(mask, dtype=torch.long).to(device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device),
            'targets': torch.tensor(self.targets[index], dtype=torch.float).to(device)
        }


# In[10]:


# Creating the dataset and dataloader for the neural network

# train_size = 0.8
# train_dataset=new_df.sample(frac=train_size,random_state=200)
# test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
# train_dataset = train_dataset.reset_index(drop=True)


# print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VAL Dataset: {}".format(val_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)


# In[11]:


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 14
                }

val_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 14
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 14
                }

training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)
testing_loader = DataLoader(testing_set, **test_params)


# In[16]:


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class BERTClass1(torch.nn.Module):
    def __init__(self):
        super(BERTClass1, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 3)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class BERTClass2(torch.nn.Module):
    def __init__(self):
        super(BERTClass2, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 768)
        self.l4 = torch.nn.Dropout(0.3)
        self.l5 = torch.nn.Linear(768, 3)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        return output


# In[17]:


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def validation(testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0), total=len(testing_loader), desc=f'validation', position=0, leave=True):
            ids = data['ids']
            mask = data['mask']
            token_type_ids = data['token_type_ids']
            targets = data['targets']
            outputs = model(ids, mask, token_type_ids)
            if len(targets) != len(outputs):
               count += 1
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_targets = torch.from_numpy(np.array(fin_targets)).float()
    fin_outputs = torch.from_numpy(np.array(fin_outputs) >= 0.5).float()
    loss = loss_fn(fin_outputs, fin_targets)
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
    return loss, accuracy, f1_score_micro


# In[18]:


dropouts = [0.3, 0.1]
lrs = [2e-05, 6.68561343998775e-5]
model_classes = [BERTClass1, BERTClass2]


# In[19]:


for dropout in dropouts:
    for lr in lrs:
        for i, model_class in enumerate(model_classes):
                        
            model = model_class()
            model.to(device)
            
            optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)
            
            wandb.init(
                project="news-nlp",
            
                config={
                    "epochs": 2,
                    "dropout": dropout,
                    "lrs": lrs,
                    "model class": i
                }
            )
            
            for epoch in range(2):
                for i,data in tqdm(enumerate(training_loader, 0), total=len(training_loader), desc=f'train epoch {epoch}'):
                    model.train()
                    ids = data['ids']
                    mask = data['mask']
                    token_type_ids = data['token_type_ids']
                    targets = data['targets']
            
                    outputs = model(ids, mask, token_type_ids)
            
                    optimizer.zero_grad()
                    loss = loss_fn(outputs, targets)
                    if i%250==0:
                        wandb.log({"training_loss": loss.item()})
                    if i%1000==0:
                        val_loss, accuracy, f1_micro = validation(val_loader)
                        wandb.log({"validation_loss": val_loss.item(),
                                    "accuracy": accuracy,
                                    "f1_micro": f1_micro})
                        
            
                    # optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # torch.save(model.state_dict(), f"./output/bert_run1/epoch{epoch}.pt")
            
            wandb.finish()


# In[30]:


model = BERTClass()
model.to(device)


# In[31]:


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


# In[32]:


optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)


# <a id='section05'></a>
# ### Fine Tuning the Model
# 
# After all the effort of loading and preparing the data and datasets, creating the model and defining its loss and optimizer. This is probably the easier steps in the process.
# 
# Here we define a training function that trains the model on the training dataset created above, specified number of times (EPOCH), An epoch defines how many times the complete data will be passed through the network.
# 
# Following events happen in this function to fine tune the neural network:
# - The dataloader passes data to the model based on the batch size.
# - Subsequent output from the model and the actual category are compared to calculate the loss.
# - Loss value is used to optimize the weights of the neurons in the network.
# - After every 5000 steps the loss value is printed in the console.
# 
# As you can see just in 1 epoch by the final step the model was working with a miniscule loss of 0.022 i.e. the network output is extremely close to the actual output.

# In[33]:


def validation(testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0), total=len(testing_loader), desc=f'validation', position=0, leave=True):
            ids = data['ids']
            mask = data['mask']
            token_type_ids = data['token_type_ids']
            targets = data['targets']
            outputs = model(ids, mask, token_type_ids)
            if len(targets) != len(outputs):
               count += 1
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_targets = torch.from_numpy(np.array(fin_targets)).float()
    fin_outputs = torch.from_numpy(np.array(fin_outputs) >= 0.5).float()
    loss = loss_fn(fin_outputs, fin_targets)
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
    return loss, accuracy, f1_score_micro


# In[34]:


wandb.init(
    project="news-nlp",

    config={
        "epochs": 5,
    }
)
wandb_flag = True


# In[ ]:


for epoch in range(2):
    for i,data in tqdm(enumerate(training_loader, 0), total=len(training_loader), desc=f'train epoch {epoch}'):
        model.train()
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        targets = data['targets']

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if i%250==0:
            wandb.log({"training_loss": loss.item()})
        if i%1000==0:
            val_loss, accuracy, f1_micro = validation(testing_loader)
            wandb.log({"validation_loss": val_loss.item(),
                        "accuracy": accuracy,
                        "f1_micro": f1_micro})
            

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # torch.save(model.state_dict(), f"./output/bert_run1/epoch{epoch}.pt")


# In[17]:


wandb.finish()
wandb_flag = False


# <a id='section06'></a>
# ### Validating the Model
# 
# During the validation stage we pass the unseen data(Testing Dataset) to the model. This step determines how good the model performs on the unseen data.
# 
# This unseen data is the 20% of `train.csv` which was seperated during the Dataset creation stage.
# During the validation stage the weights of the model are not updated. Only the final output is compared to the actual value. This comparison is then used to calcuate the accuracy of the model.
# 
# As defined above to get a measure of our models performance we are using the following metrics.
# - Accuracy Score
# - F1 Micro
# - F1 Macro
# 
# We are getting amazing results for all these 3 categories just by training the model for 1 Epoch.

# <a id='section07'></a>
# ### Saving the Trained Model Artifacts for inference
# 
# This is the final step in the process of fine tuning the model.
# 
# The model and its vocabulary are saved locally. These files are then used in the future to make inference on new inputs of news headlines.
# 
# Please remember that a trained neural network is only useful when used in actual inference after its training.
# 
# In the lifecycle of an ML projects this is only half the job done. We will leave the inference of these models for some other day.
