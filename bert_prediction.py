import pandas as pd
import numpy as np

from bert_dataset import CustomDataset
from bert_classifier import BertClassifier

import torch
import torch.nn as nn

from transformers import logging
logging.set_verbosity_error()

from transformers import BertTokenizer


# class Prediction(self):
#     def __init__(self, tokenizer, model, text):
#         self.token = tokenizer
#         self.model = model
#         self.text = text


# Path to all sources tokens:
#       /home/b4r/Documents/Classification/Models/
def predict(token_path, model_path, text):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    encoding = BertTokenizer.from_pretrained(token_path).encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print('build on', device)

    out = {
        'text': text,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()
    }

    input_ids = out["input_ids"].to(device)
    attention_mask = out["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0)
    )

    prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    return prediction
