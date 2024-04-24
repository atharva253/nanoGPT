import numpy as np
import tiktoken
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchtext.data.metrics import bleu_score
from ignite.metrics import Rouge
from transformers import AdamW

from datetime import datetime
import os
import random
from tqdm import tqdm

from model_old import GPTConfig, GPT
from data.cnn_dailymail.prepare import enc
from sentiment_config import *

import gc
import wandb
import warnings
import pandas as pd

import math

torch.cuda.empty_cache()
gc.collect()

n_layer=24
n_head=16
n_embd=1024
block_size=1024
bias=False
vocab_size=VOCAB_SIZE
dropout=dropout
EOS=VOCAB_SIZE-1
max_len=block_size
max_seq_len=block_size

# Ignore all warnings
warnings.filterwarnings("ignore")

# wandb.login(key=WANDB_KEY)


class Custom_Dataset(Dataset):
    def __init__(self, path, split,length=None):
               
        self.data =  np.load(os.path.join(path, f'{split}.npy'), mmap_mode='r')[:length]
        self.data_lens = np.load(os.path.join(path, f'{split}_lens.npy'), mmap_mode='r')[:length]
        self.labels = np.load(os.path.join(path, f'{split}_labels.npy'), mmap_mode='r')[:length]
        self.length = self.data.shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.data_lens[idx]
        lab = self.labels[idx]

        return d, l, lab


def generate_sample(model, dataset, index, out_df):
    data_sample, art_len_sample, lab = dataset[index]
    data_sample = torch.tensor(data_sample[None,:]).to(DEVICE)
    idx = art_len_sample.item()

    logits = model(data_sample, return_all_logits=True)[0]
    preds = logits[0, idx:-1, :].argmax(dim=-1).tolist()[:5]

    labels = data_sample[0, idx+1:].tolist()[:5]
    context = data_sample[0, :idx+1].tolist()
    # labels = data_sample[0, :].tolist()
    out_df.loc[i] = [enc.decode(context), enc.decode(preds), enc.decode(labels), str(lab)]

    print("Sentence:\n %s \n" % enc.decode(context))
    print("Pred Sentiment:\n %s \n" % enc.decode(preds))
    print("True Sentiment:\n %s \n\n" % enc.decode(labels))

def load_model(path):
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
    checkpoint = torch.load(path)
    #checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    #     model_args[k] = checkpoint_model_args[k]
    # create the model
    # gptconf = GPTConfig(**model_args)
    # model = GPT(gptconf)
    model = GPT.from_pretrained(init_from)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    return model.to(DEVICE)

model_summ = load_model("multitask-finetune-complete-lrdecay/bleu_ckpt.pt")
model_qa = load_model("multitask-finetune-complete-lrdecay/rouge_ckpt.pt")
print("Model Loaded!")

train_dataset = Custom_Dataset(ROOT, 'train', length=500)
print("Train Dataset Loaded!")
val_dataset = Custom_Dataset(ROOT,'validation', length=500)
print("Validation Dataset Loaded!")

## Intiialize dataloader
# train_dataloader =  DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=4)

# val_dataloader =  DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=4)

# x_train, x_lens_train, is_summarizer_train = next(iter(train_dataloader))
# x_val, x_lens_val, is_summarizer_val = next(iter(val_dataloader))

# out_train = inference(model_summ, x_train, x_lens_train, is_summarizer_train)
# out_val = inference(model_summ, x_val, x_lens_val, is_summarizer_val)

out_df1 = pd.DataFrame(columns=['sent', 'pred', 'actual', 'lab'])
out_df2 = pd.DataFrame(columns=['sent', 'pred', 'actual', 'lab'])

print("### Model 1: ###")
for i in range(100):
    generate_sample(model_summ, val_dataset, i, out_df1)

print("### Model 2: ###")
for i in range(100):
    generate_sample(model_qa, val_dataset, i, out_df2)

# print(out_train)
out_df1.to_csv("sentiment_model_1.csv")
out_df2.to_csv("sentiment_model_2.csv")



