# CHK_PT_PATH = 'pretraining-out/gpt2-medium.bin'
DEVICE = 'cuda'
OUTPUT_DIR =  'sequential-finetune-sum'
MODEL_LOAD_DIR = ''
MODEL_SAVE_DIR = 'sequential-finetune-sum'      
EPOCHS = 2
#SUMMARY_ROOT = 'data/cnn_dailymail'
#SQUAD_ROOT = 'data/squad'
ROOT = 'data/cnn_dailymail'
BATCH_SIZE = 8
IGNORE_INDEX = 50256
VOCAB_SIZE = 50257
TASK = 'SUMMARY'


# adamw optimizer
learning_rate = 6e-4 # max learning rate
# max_iters = int(EPOCHS * (150000) / BATCH_SIZE) # total number of training iterations
max_iters = 4000 # total number of training iterations
# max_iters = 100 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 240 # how many steps to warm up for
# warmup_iters = 20 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
gradient_accumulation_steps = 8 # used to simulate larger batch sizes
wandb_project = 'deep-gen-modelling'
wandb_run_name = 'sequential-finetuning-sum'
scaler_enabled = True
dropout = 0.0
init_from = "gpt2-medium"

# PAD_TOKEN = '<|pad|>'               # 50261
# SUMMARY_TOKEN = '<|summary|'        # 50258
# QUESTION_TOKEN = '<|question|>'     # 50259
# ANSWER_TOKEN = "<|answer|>"               # 50260
# SOS_TOKEN = "<|startoftext|>"       # 50257