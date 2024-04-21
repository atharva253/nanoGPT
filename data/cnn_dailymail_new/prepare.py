import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

num_proc = 8

num_proc_load_dataset = num_proc

PAD_TOKEN = '<|pad|>'               # 50261
SUMMARY_TOKEN = '<|summary|>'        # 50258
QUESTION_TOKEN = '<|question|>'     # 50259
ANSWER_TOKEN = "<|answer|>"               # 50260
SOS_TOKEN = "<|startoftext|>"       # 50257

enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.Encoding(
    name="gpt2",
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens= {
        **enc._special_tokens,
        SOS_TOKEN: 50257,
        SUMMARY_TOKEN : 50258,
        QUESTION_TOKEN : 50259,
        ANSWER_TOKEN : 50260,
        PAD_TOKEN : 50261
    }
)

if __name__ == '__main__':
    dataset = load_dataset("cnn_dailymail", '3.0.0', num_proc=num_proc_load_dataset)

    def process(example):
        article_ids = [50257] + enc.encode_ordinary(example['article']) # encode_ordinary ignores any special tokens 

        highlights_ids = enc.encode_ordinary(example['highlights'])
        highlights_ids.append(enc.eot_token)

        input_ids = article_ids + [50258]

        data = input_ids + highlights_ids
        if len(data) > 1024:
            text = [0]
        else:
            # Pad sequences to length 1024
            text = [50261]*1024
            text[:len(data)] = data

        out = {'data': text, 'data_len': len(data), 'article_lens': [len(input_ids)-1]}

        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['article','highlights','id'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Drop examples with input sequence lengths < 1024
    tokenized['train'] = tokenized['train'].filter(lambda data: len(data['data']) == 1024)
    tokenized['validation'] = tokenized['validation'].filter(lambda data: len(data['data']) == 1024)
    tokenized['test'] = tokenized['test'].filter(lambda data: len(data['data']) == 1024)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        filename = os.path.join(os.path.dirname(__file__), f'{split}')
        np.save(filename, np.array(dset['data']))


        filename = os.path.join(os.path.dirname(__file__), f'{split}_lens')
        np.save(filename, np.array(dset['article_lens']))