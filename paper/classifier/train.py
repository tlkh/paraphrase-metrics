import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-m", type=str, default="microsoft/deberta-large")
parser.add_argument("-d", type=str, default="mrpc")
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-bs", type=int, default=32)
parser.add_argument("-warmup", type=float, default=0.4)
parser.add_argument("-log", type=float, default=50)
parser.add_argument("-gradaccum", type=int, default=1)

args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SEED = args.seed
import random
random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)
import torch
torch.manual_seed(args.seed)
import csv
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

def preprocess_text(input_text):
    input_text = str(input_text).strip()
    input_text = input_text.replace("``", "''").replace("‘‘", '"').replace("’’", '"').replace("''", '"')
    input_text = input_text.replace("[", "").replace("]", "")
    input_text = input_text.replace(" .", ".").replace(" ,", ",")
    input_text = input_text.replace("’", "'").replace("“", '"').replace("”", '"')
    return input_text.replace("  ", " ")

MAX_SEQUENCE_LENGTH = 128
MODEL_NAME = args.m
TRAIN_DATASET = args.d

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_text(t1, t2):
    o = tokenizer(t1, t2,
                  padding="max_length",
                  truncation="longest_first",
                  max_length=MAX_SEQUENCE_LENGTH,
                  return_token_type_ids=True,
                  return_attention_mask=True)
    return o

train_sent_1_list, train_sent_2_list, train_labels_raw = [], [], []

if TRAIN_DATASET=="PAWS":
    with open("../datasets/paws/train.tsv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        for row in reader:
            label = int(row[3])
            train_sent_1_list.append(preprocess_text(row[1]))
            train_sent_2_list.append(preprocess_text(row[2]))
            train_labels_raw.append(label)
if TRAIN_DATASET=="MRPC":
    with open("../datasets/mrpc/mrpc_train.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for row in reader:
            label = int(row[3])
            train_sent_1_list.append(preprocess_text(row[1]))
            train_sent_2_list.append(preprocess_text(row[2]))
            assert len(row[1]) > 5
            train_labels_raw.append(label)
if TRAIN_DATASET=="MRPC_COR":
    with open("../datasets/mrpc_train_corrected.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for row in reader:
            label = int(row[-1])
            train_sent_1_list.append(preprocess_text(row[0]))
            train_sent_2_list.append(preprocess_text(row[1]))
            assert len(row[1]) > 5
            train_labels_raw.append(label)

train_tokens = []
train_type_ids = []
train_attn_masks = []
train_labels = []

for t1, t2, l in zip(train_sent_1_list, train_sent_2_list, train_labels_raw):
    t = tokenize_text(t1, t2)
    train_tokens.append(t["input_ids"])
    train_type_ids.append(t["token_type_ids"])
    train_attn_masks.append(t["attention_mask"])
    train_labels.append(l)
    t = tokenize_text(t2, t1)
    train_tokens.append(t["input_ids"])
    train_type_ids.append(t["token_type_ids"])
    train_attn_masks.append(t["attention_mask"])
    train_labels.append(l)
    
train_tokens = np.asarray(train_tokens)
train_type_ids = np.asarray(train_type_ids)
train_attn_masks = np.asarray(train_attn_masks)
train_labels = np.asarray(train_labels)

valid_sent_1_list, valid_sent_2_list, valid_labels_raw = [], [], []

with open("../datasets/paws/dev.tsv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    next(reader, None)  # skip the headers
    for row in reader:
        label = int(row[3])
        valid_sent_1_list.append(preprocess_text(row[1]))
        valid_sent_2_list.append(preprocess_text(row[2]))
        valid_labels_raw.append(label)
    
valid_tokens = []
valid_type_ids = []
valid_attn_masks = []
valid_labels = []

for t1, t2, l in zip(valid_sent_1_list, valid_sent_2_list, valid_labels_raw):
    t = tokenize_text(t1, t2)
    valid_tokens.append(t["input_ids"])
    valid_type_ids.append(t["token_type_ids"])
    valid_attn_masks.append(t["attention_mask"])
    valid_labels.append(l)
    
valid_tokens = np.asarray(valid_tokens)
valid_type_ids = np.asarray(valid_type_ids)
valid_attn_masks = np.asarray(valid_attn_masks)
valid_labels = np.asarray(valid_labels)

test_sent_1_list, test_sent_2_list, test_labels_raw = [], [], []
        
with open("../datasets/paws/test.tsv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    next(reader, None)  # skip the headers
    for row in reader:
        label = int(row[3])
        test_sent_1_list.append(preprocess_text(row[1]))
        test_sent_2_list.append(preprocess_text(row[2]))
        test_labels_raw.append(label)

test_tokens = []
test_type_ids = []
test_attn_masks = []
test_labels = []

for t1, t2, l in zip(test_sent_1_list, test_sent_2_list, test_labels_raw):
    t = tokenize_text(t1, t2)
    test_tokens.append(t["input_ids"])
    test_type_ids.append(t["token_type_ids"])
    test_attn_masks.append(t["attention_mask"])
    test_labels.append(l)
    
test_tokens = np.asarray(test_tokens)
test_type_ids = np.asarray(test_type_ids)
test_attn_masks = np.asarray(test_attn_masks)
test_labels = np.asarray(test_labels)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, type_ids, attn_masks, labels):
        self.tokens = tokens
        self.type_ids = type_ids
        self.attn_masks = attn_masks
        self.labels = labels
    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.tokens[idx], dtype=torch.int64),
            "token_type_ids": torch.tensor(self.type_ids[idx], dtype=torch.int64),
            "attention_mask": torch.tensor(self.attn_masks[idx], dtype=torch.int64),
        }
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_tokens, train_type_ids, train_attn_masks, train_labels)
valid_dataset = TextDataset(valid_tokens, valid_type_ids, valid_attn_masks, valid_labels)
test_dataset = TextDataset(test_tokens, test_type_ids, test_attn_masks, test_labels)

d0 = train_dataset[0]
print(tokenizer.decode(d0["input_ids"]))
print(d0["labels"])

d0 = valid_dataset[0]
print(tokenizer.decode(d0["input_ids"]))
print(d0["labels"])

d0 = test_dataset[0]
print(tokenizer.decode(d0["input_ids"]))
print(d0["labels"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

s = str(args.seed)
d = TRAIN_DATASET.lower()
model_name_short = MODEL_NAME.split("/")[-1]

LOGGING_STEP = args.log
training_args = TrainingArguments(
    output_dir="./results/"+d+"/"+model_name_short+"/output_"+s+"/",
    logging_dir="./results/"+d+"/"+model_name_short+"/log_"+s+"/",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=256,
    gradient_accumulation_steps=args.gradaccum,
    weight_decay=0.00,
    warmup_ratio=args.warmup,
    lr_scheduler_type="linear",
    eval_steps=LOGGING_STEP,
    logging_steps=LOGGING_STEP,
    save_steps=LOGGING_STEP,
    evaluation_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    save_total_limit=4,
    learning_rate=5e-6,
    fp16=True,
    fp16_full_eval=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    disable_tqdm=True,
    load_best_model_at_end=True
)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=valid_dataset,
                  compute_metrics=compute_metrics,
                  callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=4,
                                                                early_stopping_threshold=0.01),])
trainer.remove_callback(transformers.integrations.TensorBoardCallback)

trainer.train()

os.system("rm -rf /results/"+d+"/"+model_name_short+"/output_"+s+"/checkpoint-*")

metrics = trainer.evaluate(metric_key_prefix="valid")
trainer.log_metrics("valid", metrics)
trainer.save_metrics("valid", metrics)

metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)
