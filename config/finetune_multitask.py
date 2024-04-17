import time

out_dir = 'multitask-finetune-complete'
eval_interval = 1000
eval_iters = 200
log_interval = 10

wandb_log = True # feel free to turn on
wandb_project = 'multitask-finetune'
wandb_run_name = 'multitask-finetune-' + str(time.time())

dataset = 'mixed'
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

max_iters = 50000
lr_decay_iters = 50000

warmup_iters = 200
learning_rate = 6e-4
min_lr = 6e-5
decay_lr = True
weight_decay = 1e-1


beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

scaler_enabled = True
dropout = 0.0