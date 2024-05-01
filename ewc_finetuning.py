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
import pickle
from tqdm import tqdm
import json
from model_old import GPTConfig, GPT
from data.cnn_dailymail.prepare import enc
from ewc_config import *

import gc
import wandb
import warnings

import math

torch.cuda.empty_cache()
gc.collect()

n_layer=12
n_head=12
n_embd=768
block_size=1024
bias=False
vocab_size=None
dropout=dropout
ewc_lambda = 0.1

# enc = tiktoken.get_encoding("gpt2")
# enc = tiktoken.Encoding(
#     name="gpt2",
#     pat_str=enc._pat_str,
#     mergeable_ranks=enc._mergeable_ranks,
#     special_tokens={
#         **enc._special_tokens,
#         SOS_TOKEN: 50257,
#         SUMMARY_TOKEN : 50258,
#         QUESTION_TOKEN : 50259,
#         ANSWER_TOKEN : 50260,
#         PAD_TOKEN : 50261
#     }
# )

# Ignore all warnings
warnings.filterwarnings("ignore")

# wandb.login(key=WANDB_KEY)

## Dataset Class
class Custom_Dataset(Dataset):
    def __init__(self, path, file, length=None):
        # # Merge the datasets for the summarizer and QA tasks
        # self.summarise_data = np.load(os.path.join(summary_root, file+'.npy'), mmap_mode='r')[:length]
        # self.summarise_lens = np.load(os.path.join(summary_root, file+'_lens.npy'), mmap_mode='r')[:length]

        # self.qa_data = np.load(os.path.join(squad_root, file+'.npy'), mmap_mode='r')[:length]
        # self.qa_lens = np.load(os.path.join(squad_root, file+'_lens.npy'), mmap_mode='r')[:length]

        # # self.summarise_data = np.load(os.path.join(summary_root, file+'.npy'), mmap_mode='r')
        # # self.summarise_lens = np.load(os.path.join(summary_root, file+'_lens.npy'), mmap_mode='r')

        # # self.qa_data = np.load(os.path.join(squad_root, file+'.npy'), mmap_mode='r')
        # # self.qa_lens = np.load(os.path.join(squad_root, file+'_lens.npy'), mmap_mode='r')

        # self.data = np.concatenate([self.summarise_data, self.qa_data])
        # self.data_lens = np.concatenate([self.summarise_lens, self.qa_lens])

        # self.data =  np.load(os.path.join(path, file+'.npy'), mmap_mode='r')[:length]
        # self.data_lens = np.load(os.path.join(path, file+'_lens.npy'), mmap_mode='r')[:length]

        # self.length = self.data.shape[0]
        self.file = file
        #self.task = TASK
        

        if file=='train': # train data

            self.data =  np.load(os.path.join(path, file+'.npy'), mmap_mode='r')[:length]
            self.data_lens = np.load(os.path.join(path, file+'_lens.npy'), mmap_mode='r')[:length]
            self.length = self.data.shape[0]


        else:   # validation data

            self.summarise_data = np.load(os.path.join(path,'..','cnn_dailymail', file+'.npy'), mmap_mode='r')[:length]
            self.summarise_lens = np.load(os.path.join(path,'..','cnn_dailymail', file+'_lens.npy'), mmap_mode='r')[:length]

            self.qa_data = np.load(os.path.join(path,'..','squad', file+'.npy'), mmap_mode='r')[:length]
            self.qa_lens = np.load(os.path.join(path,'..','squad', file+'_lens.npy'), mmap_mode='r')[:length]

            self.data = np.concatenate([self.summarise_data, self.qa_data])
            self.data_lens = np.concatenate([self.summarise_lens, self.qa_lens])


            self.length = self.data.shape[0]





    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.data_lens[idx]

        if self.file=='train':
            return d, l
        else: 
            return d, l,  idx<len(self.summarise_data)


train_dataset = Custom_Dataset(ROOT, 'train', length=75000)
print("Train Dataset Loaded!")
val_dataset = Custom_Dataset(ROOT,'validation', length=5000)
print("Validation Dataset Loaded!")

# Loading train dataset of task1-summarizer
task1_train_dataset = Custom_Dataset(TASK1_ROOT, 'train', length=75000)
print("Task 1 Train Dataset Loaded!")


## Intiialize dataloader
train_dataloader =  DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4)

val_dataloader =  DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4)

task1_train_dataloader =  DataLoader(
    task1_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4)



# ## Load pretrained model ##
# checkpoint = torch.load(CHK_PT_PATH, map_location=DEVICE)
# checkpoint_model_args = checkpoint['model_args']
# checkpoint_model_args['dropout'] = dropout

# gptconf = GPTConfig(**checkpoint_model_args)
# model = GPT(gptconf)
# state_dict = checkpoint['model']

# unwanted_prefix = '_orig_mod.'
# for k,v in list(state_dict.items()):
#     if k.startswith(unwanted_prefix):
#         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
# model.load_state_dict(state_dict)
# iter_num = checkpoint['iter_num']
# best_val_loss = checkpoint['best_val_loss']

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'resume':
    print(f"Resuming training from {MODEL_LOAD_DIR}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(MODEL_LOAD_DIR, 'bleu_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        #checkpoint_model_args = checkpoint['model_args']  # not using this since we did not save model_args in the first model
    override_args = dict(dropout=dropout)
    original_model = GPT.from_pretrained("gpt2-medium", override_args)
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        #model_args[k] = checkpoint_model_args[k]   # not using this since we did not save model_args in the first model 
        model_args[k] = getattr(original_model.config, k)
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
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
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(DEVICE)


# Setup training functions
loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
val_loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

## Init wandb
# wandb.login(key=WANDB_KEY)
wandb_config = {
    'BATCH_SIZE': BATCH_SIZE,
    'learning_rate': learning_rate,
    'gradient_accumulation_steps': gradient_accumulation_steps
}
wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

model = model.to(DEVICE)

name = ''
for temp, param in model.named_parameters():
    name = temp
    print(name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# ewc_prep() should be called after the training of first task is complete, using the dataloader of the first task
def ewc_prep(num_sample=5000):

    # some resources use eval() instead
    # model.eval()
    model.train()
    optimizer.zero_grad()

    # accumulating gradients

    # the implementation could be buggy without actually running the code, basically it's the same as the training process BUT
    # 1. call optimizer.zero_grad() only BEFORE instead of WITHIN the iteration loop; 
    # 2. no calling to optimizer.step() or mixed-precision equivalent version after loss.backward()

    for step, (data, article_len) in enumerate(tqdm(task1_train_dataloader)):
        inputs, labels = torch.tensor(data), torch.tensor(data)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(inputs, return_all_logits=True)[0]
        
        # only consider loss on reference summary just like seq2seq models
        shift_logits = []
        shift_labels = []
        for batch_idx in range(logits.shape[0]):
            idx = article_len[batch_idx].item() # index of separator token
            shift_logits.append(logits[batch_idx, idx:-1, :])
            shift_labels.append(labels[batch_idx, idx+1:])
        shift_logits = torch.cat(shift_logits, dim=0)
        shift_labels = torch.cat(shift_labels, dim=0)
        
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss/gradient_accumulation_steps

        # not sure which one should be used here
        # loss.backward()
        scaler.scale(loss).backward()

        del inputs, labels
        torch.cuda.empty_cache()
        gc.collect()
        if(step*8 > num_sample):
            break

    fisher_dict = {}
    optpar_dict = {}

    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
        optpar_dict[name] = param.data.clone()
        fisher_dict[name] = param.grad.data.clone().pow(2)
    
    return fisher_dict, optpar_dict

# fisher_dict and optpar_dict is fixed and not updated during training, could declare them outside as global variables
def train_ewc(model, fisher_dict, optpar_dict):
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(1337)
    model.train()
    for epoch in np.arange(EPOCHS):
        # is_summariser = TASK
        for step, (data, article_len) in enumerate(tqdm(train_dataloader)):
            # print('before anything')
            # print(torch.cuda.memory_summary(abbreviated=False))
            inputs, labels = torch.tensor(data), torch.tensor(data)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # model.train()
            # print('before train inf')
            # print(torch.cuda.memory_summary(abbreviated=False))
            # with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(inputs, return_all_logits=True)[0]
                # continue
                # torch.cuda.empty_cache()
                # gc.collect()
                # print('after train inf')
                # print(torch.cuda.memory_summary(abbreviated=False))
                
                # only consider loss on reference summary just like seq2seq models
                shift_logits = []
                shift_labels = []
                for batch_idx in range(logits.shape[0]):
                    idx = article_len[batch_idx].item() # index of separator token
                    shift_logits.append(logits[batch_idx, idx:-1, :])
                    shift_labels.append(labels[batch_idx, idx+1:])
                shift_logits = torch.cat(shift_logits, dim=0)
                shift_labels = torch.cat(shift_labels, dim=0)

                # print('before loss')
                # print(torch.cuda.memory_summary(abbreviated=False))
                loss = loss_fct(shift_logits, shift_labels)/gradient_accumulation_steps
                # print('after loss')
                # print(torch.cuda.memory_summary(abbreviated=False))
            
                # ewc change starts
                # 2 prints are for adjusting hyperparameter ewc_lambda

                print('loss before ewc: {}'.format(loss.item()))
                # counter = 0
                # for name, param in reversed(model.named_parameters()):
                #     # fisher = fisher_dict[name]
                #     # optpar = optpar_dict[name]
                #     # hyperparameter ewc_lambda 
                #     # ewc_loss = torch.tensor()
                #     loss = loss + (fisher_dict[name] * (optpar_dict[name] - param).pow(2)).sum() * ewc_lambda
                #     break
                loss = loss + (fisher_dict[name] * (optpar_dict[name] - param).pow(2)).sum() * ewc_lambda
                print('loss after ewc: {}'.format(loss.item()))
                # ewc change ends

            # loss = loss/gradient_accumulation_steps
            # torch.cuda.empty_cache()
            # gc.collect()
            # print('before loss back')
            # print(torch.cuda.memory_summary(abbreviated=False))
            scaler.scale(loss).backward()
            # print('after loss back')
            # print(torch.cuda.memory_summary(abbreviated=False))

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                # print('optimize stuff')
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # gradient clipping overcomes sum/mean in CrossEntropy

                scaler.step(optimizer)
                scaler.update()
               
                model.zero_grad()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                logging_loss = tr_loss
                print("loss:", loss.item(), end='\n\n')
                
                if (step + 1)/gradient_accumulation_steps == 1.0:
                    print('After 1st update: ', end='\n\n')
                    generate_sample(0)
                    generate_sample(-5)
                else:
                    print('After', global_step+1,'updates: ', end='\n\n')
                    generate_sample(0)
                    generate_sample(-5)
            
                if (step + 1) % (10*gradient_accumulation_steps) == 0:
                    results = evaluate(model, global_step, lr, loss.item())        
                
            
            del inputs, labels, logits, idx, shift_logits, shift_labels
            torch.cuda.empty_cache()
            gc.collect()
            if (global_step * 8) >= max_iters: break

def train(model):
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(1337)
    for epoch in np.arange(EPOCHS):
        # is_summariser = TASK
        for step, (data, article_len) in enumerate(tqdm(train_dataloader)):
            inputs, labels = torch.tensor(data), torch.tensor(data)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            model.train()
            logits = model(inputs, return_all_logits=True)[0]
            
            # only consider loss on reference summary just like seq2seq models
            shift_logits = []
            shift_labels = []
            for batch_idx in range(logits.shape[0]):
                idx = article_len[batch_idx].item() # index of separator token
                shift_logits.append(logits[batch_idx, idx:-1, :])
                shift_labels.append(labels[batch_idx, idx+1:])
            shift_logits = torch.cat(shift_logits, dim=0)
            shift_labels = torch.cat(shift_labels, dim=0)

            loss = loss_fct(shift_logits, shift_labels)
            loss = loss/gradient_accumulation_steps
            scaler.scale(loss).backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # gradient clipping overcomes sum/mean in CrossEntropy

                scaler.step(optimizer)
                scaler.update()
               
                model.zero_grad()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                logging_loss = tr_loss
                print("loss:", loss.item(), end='\n\n')
                
                if (step + 1)/gradient_accumulation_steps == 1.0:
                    print('After 1st update: ', end='\n\n')
                    generate_sample(0)
                    generate_sample(-5)
                else:
                    print('After', global_step+1,'updates: ', end='\n\n')
                    generate_sample(0)
                    generate_sample(-5)
            
                if (step + 1) % (10*gradient_accumulation_steps) == 0:
                    results = evaluate(model, global_step, lr, loss.item())        
                
            
            del inputs, labels
            torch.cuda.empty_cache()
            gc.collect()
            if (global_step * 8) >= max_iters: break

def generate_sample(index):
    data_sample, art_len_sample,is_summariser = val_dataset[index]
    data_sample = torch.tensor(data_sample[None,:]).to(DEVICE)
    idx = art_len_sample.item()

    logits = model(data_sample, return_all_logits=True)[0]
    preds = logits[0, idx:-1, :].argmax(dim=-1).tolist()

    labels = data_sample[0, idx+1:].tolist()

    if index == 0:
        print("Pred Summary:\n %s \n" % enc.decode(preds))
        print("True Summary:\n %s \n\n" % enc.decode(labels))
    else:
        print("Pred Answer:\n %s \n" % enc.decode(preds))
        print("True Answer:\n %s \n\n" % enc.decode(labels))
    
    del data_sample



def evaluate(model, global_step=None, lr=None, tr_loss=None):
    if not os.path.exists(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)
    eval_output_dir = MODEL_SAVE_DIR

    results = {}

    eval_loss = 0.0
    eval_bleu_scores = 0.0
    eval_rouge_scores = 0.0
    nb_eval_steps = 0
    model.eval()

    for (data, article_len,is_summariser) in tqdm(val_dataloader):
        inputs, labels = torch.tensor(data).to(DEVICE), torch.tensor(data).to(DEVICE)
        with torch.no_grad():
            logits = model(inputs, return_all_logits=True)[0]
            shift_logits = []
            shift_labels = []
            avg_eval_bleu = 0.0
            avg_rouge_score = 0.0
            # m = Rouge(variants=["L",1,2], multiref="best")
            m = Rouge(variants=["L"], multiref="best")

            for batch_idx in range(logits.shape[0]):
                idx = article_len[batch_idx].item() # index of separator token
                
                shift_logits.append(logits[batch_idx, idx:-1, :])
                shift_labels.append(labels[batch_idx, idx+1:])
                
                greedy_labels = labels[batch_idx, idx+1:].tolist()
                index = greedy_labels.index(enc.eot_token)
                greedy_labels  = greedy_labels[:index]
                references = [[enc.decode(greedy_labels).split()]]

                greedy_preds = logits[batch_idx, idx:-1, :].argmax(dim=-1).tolist()
                greedy_preds = greedy_preds[:index]
                hypotheses = [enc.decode(greedy_preds).split()]

                if is_summariser[batch_idx].item():
                    bleu4 = bleu_score(hypotheses, references, max_n=2, weights=[0.5, 0.5])
                    avg_eval_bleu += bleu4
                else:
                    m.update((hypotheses, references))
                    rouge = m.compute()
                    # avg_rouge_score += max(rouge.values())
                    avg_rouge_score += rouge['Rouge-L-F']

            
            shift_logits = torch.cat(shift_logits, dim=0)
            shift_labels = torch.cat(shift_labels, dim=0)
            
            lm_loss = loss_fct(shift_logits, shift_labels)
            eval_loss += lm_loss.mean().item()

            eval_bleu_scores += avg_eval_bleu/logits.shape[0]
            eval_rouge_scores += avg_rouge_score/logits.shape[0]
        
        del inputs, labels, logits, shift_logits, shift_labels
        torch.cuda.empty_cache()
        gc.collect()


        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_bleu_scores = 2*eval_bleu_scores / nb_eval_steps
    eval_rouge_scores = 2*eval_rouge_scores / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity,
        'eval_bleu_scores': eval_bleu_scores,
        'eval_rouge_scores': eval_rouge_scores
    }
    print("perplexity:", perplexity.item())
    print('eval_bleu_scores: ', eval_bleu_scores)
    print('eval_rouge_scores: ', eval_rouge_scores)

    global best_bleu_score
    global best_rouge_score
    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, step = %s\n" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))

        wandb.log({
                "iter": global_step,
                "train/loss": tr_loss,
                "val/loss": eval_loss,
                'eval_bleu_scores': eval_bleu_scores,
                'eval_rouge_scores': eval_rouge_scores,
                "lr": lr,
            })

        if eval_bleu_scores >= best_bleu_score:
            best_bleu_score = eval_bleu_scores
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'model_args': checkpoint_model_args,
                'iter_num': global_step,
                'best_val_loss': min(best_val_loss, eval_loss),
                'best_bleu_score': best_bleu_score,
                'best_rouge_score': best_rouge_score,
                # 'config': gptconf,
            }
            print(f"saving checkpoint to {eval_output_dir}")
            torch.save(checkpoint, os.path.join(eval_output_dir, 'bleu_ckpt.pt'))
        
        if eval_rouge_scores >= best_rouge_score:
            best_rouge_score = eval_rouge_scores
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'model_args': checkpoint_model_args,
                'iter_num': global_step,
                'best_val_loss': min(best_val_loss, eval_loss),
                'best_bleu_score': best_bleu_score,
                'best_rouge_score': best_rouge_score,
                # 'config': gptconf,
            }
            print(f"saving checkpoint to {eval_output_dir}")
            torch.save(checkpoint, os.path.join(eval_output_dir, 'rouge_ckpt.pt'))

    return result 

best_bleu_score = 0.21
best_rouge_score = 0.43
best_val_loss = 10000

# train(model)
# fisher_dict, optpar_dict = ewc_prep(num_sample=10000)
# # File path to save the dictionary
# file_path = 'fisher_dict.pkl'

# # Save dictionary to file using pickle
# with open(file_path, 'wb') as f:
#     pickle.dump(fisher_dict, f)

# # File path to save the dictionary
# file_path = 'optpar_dict.pkl'

# # Save dictionary to file using pickle
# with open(file_path, 'wb') as f:
#     pickle.dump(optpar_dict, f)

with open('fisher_dict.pkl', 'rb') as f:
    fisher_dict = pickle.load(f)
with open('optpar_dict.pkl', 'rb') as f:
    optpar_dict = pickle.load(f)
print("Dictionary saved successfully!")
print("Dictionaries created")
# first task complete
# model/dataset loading

torch.cuda.empty_cache()
gc.collect()
train_ewc(model, fisher_dict, optpar_dict)
