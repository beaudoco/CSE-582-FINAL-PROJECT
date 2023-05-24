#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! pip install transformers datasets
# ! pip install transformers datasets evaluate bleu


# In[2]:


import re
import torch
import string
import pandas
import random
import evaluate
import unicodedata
import numpy as np
from io import open
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline


# In[3]:

#Set seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.cuda.empty_cache()

checkpoint = "t5-small"
folder = "/scratch/ivm5230/collin/sindarin-nmt/t5-small-1ep"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
source_lang = "en"
# target_lang = "fr"
target_lang = "si"
prefix = "translate English to Sindarin: "
metric = evaluate.load("sacrebleu")
batch_size = 1
# metric = evaluate.load("bleu")

# notebook_login()


# In[4]:


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=64, truncation=True)
    return model_inputs


# In[5]:


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# In[6]:


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# In[7]:


data = pandas.read_csv('data/sindarin-eng.txt' , sep='\t', lineterminator='\n')
data = data.dropna().drop_duplicates()
data = data.reset_index()
data.columns = ['id','si', 'en']

data = data.replace({r'\r': ''}, regex=True)
data = data.replace({r'[^\w\s]': ''}, regex=True)

data['si'] = data['si'].str.lower()
data['en'] = data['en'].str.lower()

data['si'] = data['si'].str.strip()
data['en'] = data['en'].str.strip()

data['si'] = data['si'].str.normalize('NFD').str.encode('ascii', errors='ignore').str.decode('utf-8')
data['en'] = data['en'].str.normalize('NFD').str.encode('ascii', errors='ignore').str.decode('utf-8')

data = data.dropna().drop_duplicates()

data['translation'] = data[['si', 'en']].apply(dict, axis=1)
data.drop(['si', 'en'], axis=1, inplace=True)

print(data)


# In[8]:


# sind = load_dataset('text', data_files={'train': 'data/sindarin-eng.txt'})
# train_ds = Dataset.from_pandas(data)
# sind['train'] = train_ds
# sind = load_dataset({'train': dataset_train})
# sind = sind.remove_columns(["__index_level_0__"])
# print(sind['train'][0])


# In[9]:


# books = load_dataset("opus_books", "en-fr")
# books = books["train"].train_test_split(test_size=0.2)
# print(books["train"][0])

ds = Dataset.from_pandas(data)
# Create the test set
ds = ds.train_test_split(test_size=0.2, seed=42)
test_ds = ds['test']
# Create the validation set
ds = ds['train'].train_test_split(test_size=0.2, seed=42)
train_ds, valid_ds = ds['train'], ds['test']
# Put train, validation and test into a DatasetDict
ds = DatasetDict({'train': train_ds, 'valid': valid_ds, 'test': test_ds})

tokenized_books = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

torch.cuda.empty_cache()

training_args = Seq2SeqTrainingArguments(
    output_dir= folder,
    evaluation_strategy="steps",
    eval_steps=500,
    optim='adafactor',
    learning_rate=5e-5,
    warmup_steps=1000,
    lr_scheduler_type='linear',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=500,
    max_steps=5000,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# In[10]:


trainer.train()
# trainer.push_to_hub()


# In[ ]:


text = "translate English to Sindarin: who is Aragorn"

translator = pipeline("translation", model= folder)
translator(text)

tokenizer = AutoTokenizer.from_pretrained(folder)
inputs = tokenizer(text, return_tensors="pt").input_ids

model = AutoModelForSeq2SeqLM.from_pretrained(folder)
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
tokenizer.decode(outputs[0], skip_special_tokens=True)

