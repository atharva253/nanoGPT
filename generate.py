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
from finetune_config_old import *

import gc
import wandb
import warnings

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


class MergedDataset(Dataset):
    def __init__(self, summary_root, squad_root, file, length=None):
        # Merge the datasets for the summarizer and QA tasks
        self.summarise_data = np.load(os.path.join(summary_root, file+'.npy'), mmap_mode='r')[:length]
        self.summarise_lens = np.load(os.path.join(summary_root, file+'_lens.npy'), mmap_mode='r')[:length]

        self.qa_data = np.load(os.path.join(squad_root, file+'.npy'), mmap_mode='r')[:length]
        self.qa_lens = np.load(os.path.join(squad_root, file+'_lens.npy'), mmap_mode='r')[:length]

        # self.summarise_data = np.load(os.path.join(summary_root, file+'.npy'), mmap_mode='r')
        # self.summarise_lens = np.load(os.path.join(summary_root, file+'_lens.npy'), mmap_mode='r')

        # self.qa_data = np.load(os.path.join(squad_root, file+'.npy'), mmap_mode='r')
        # self.qa_lens = np.load(os.path.join(squad_root, file+'_lens.npy'), mmap_mode='r')

        self.data = np.concatenate([self.summarise_data, self.qa_data])
        self.data_lens = np.concatenate([self.summarise_lens, self.qa_lens])

        self.length = self.data.shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.data_lens[idx]

        return d, l, idx<len(self.summarise_data)


def remove_pad(sent):
    '''truncate the sentence if BOS is in it,
     otherwise simply remove the padding tokens at the end'''
    if sent.count(EOS)>0:
      sent = sent[0:sent.index(EOS)+1]
    while sent and sent[-1] == EOS:
            sent = sent[:-1]
    return sent

def decode_sentence(detokenizer, sentence_ids):
    'convert a tokenized sentence (a list of numbers) to a literal string'
    if not isinstance(sentence_ids, list):
        sentence_ids = sentence_ids.tolist()
    sentence_ids = remove_pad(sentence_ids)
    return detokenizer(sentence_ids).replace("", "")\
           .replace("", "").strip().replace(" .", ".")

def inference(model, x, x_len, is_summarizer):
    'translate source sentences into the target language, without looking at the answer'
    with torch.no_grad():
        # TODO: implement the rest of it. Encode the src sentence and then iteratively decode the target tokens
        bsz = x.size(0)
        y_list = []
        for i in range(bsz):
          y = torch.tensor([remove_pad(x[i,:x_len[i]].tolist())], dtype=torch.int32).to(DEVICE)
          for i in range(max_seq_len-2):
            pred = model(y, return_all_logits=True)[0]
            print(pred.shape)
            out = torch.argmax(pred,dim=-1).view(1, 1)
            y = torch.cat([y,out], dim=1)
            if out.item() == EOS:
              break
          y_list.append(torch.squeeze(y))

    return enc.decode(y_list)

def generate_sample(model, dataset, index):
    data_sample, art_len_sample, _ = dataset[index]
    data_sample = torch.tensor(data_sample[None,:]).to(DEVICE)
    idx = art_len_sample.item()

    logits = model(data_sample, return_all_logits=True)[0]
    preds = logits[0, idx:-1, :].argmax(dim=-1).tolist()

    labels = data_sample[0, idx+1:].tolist()
    context = data_sample[0, :idx+1].tolist()
    # labels = data_sample[0, :].tolist()
    
    print("Context:\n %s \n" % enc.decode(context))
    if index < 5000:
        print("Pred Summary:\n %s \n" % enc.decode(preds))
        print("True Summary:\n %s \n\n" % enc.decode(labels))
    else:
        print("Pred Answer:\n %s \n" % enc.decode(preds))
        print("True Answer:\n %s \n\n" % enc.decode(labels))

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

train_dataset = MergedDataset(SUMMARY_ROOT, SQUAD_ROOT, 'train', length=75000)
print("Train Dataset Loaded!")
val_dataset = MergedDataset(SUMMARY_ROOT, SQUAD_ROOT,'validation', length=5000)
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

print("### Model 1: ###")
for i in range(50):
    generate_sample(model_summ, val_dataset, i)

for i in range(50):
    generate_sample(model_summ, val_dataset, 5000+i)

print("### Model 2: ###")
for i in range(50):
    generate_sample(model_qa, val_dataset, i)

for i in range(50):
    generate_sample(model_qa, val_dataset, 5000+i)

# print(out_train)


