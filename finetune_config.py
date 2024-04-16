CHK_PT_PATH = 'pretraining-out/gpt2-medium.bin'
DEVICE = 'cuda'
OUTPUT_DIR = 'multitask-finetune-complete'
EPOCHS = 2
SUMMARY_ROOT = 'data/cnn_dailymail'
SQUAD_ROOT = 'data/squad'
BATCH_SIZE = 16
IGNORE_INDEX = -1
VOCAB_SIZE = 50257
WANDB_KEY = "2e553daaf6ad9ea48b51a7ba8f0f08fff07525b7"

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 50000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = 50000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
gradient_accumulation_steps = 8 # used to simulate larger batch sizes
wandb_project = 'deep-gen-modelling'
wandb_run_name = 'multitask-finetuning'
scaler_enabled = True
dropout = 0.0
init_from = "gpt2-medium"