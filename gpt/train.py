import torch
import os
import numpy as np
import sys 
import yaml
import wandb
import time

from model import GPT

# Get arguments
args = yaml.safe_load(open(sys.argv[1], 'r'))
data_args = args.get("data", {})
batch_size = data_args.get("batch_size", 32)
model_args = args.get("model", {})
block_size = model_args.get("block_size", 1024)
optimization_args = args.get("optimization", {})
device = optimization_args.get("device", "cuda:0")

# Load dataset
dataset = data_args.get("dataset", "shakespeare")
train_file = os.path.join(os.path.dirname(__file__), "data", dataset, "train.bin")
val_file = os.path.join(os.path.dirname(__file__), "data", dataset, "val.bin")
train_data = np.memmap(train_file, mode='r', dtype=np.uint16)
val_data = np.memmap(val_file, mode='r', dtype=np.uint16)

# init wandb
wandb_project = 'gpt'
run_name = optimization_args.get('run_name', 'generic_run')
wandb.init(project=wandb_project, name=run_name, config=args)

# get batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_indexes = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = torch.stack([
        torch.from_numpy(data[si:si+block_size].astype(np.int64))
        for si in start_indexes
    ])
    y = torch.stack([
        torch.from_numpy(data[si + 1:si + 1 + block_size].astype(np.int64))
        for si in start_indexes
    ])
    x, y = x.to(device), y.to(device)
    return x, y

# estimate loss
@torch.no_grad()
def estimate_loss(model, eval_iters=100):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x,y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out 

# set up optimization loop
iter_num = 0
learning_rate = float(optimization_args.get("lr", 3e-4))
max_iters = 4000000
eval_interval = 100
out_dir = "./out/"
best_loss = 1e9

model = GPT(**model_args)
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
while True: 
    x, y = get_batch('train')
    logits, loss = model(x, y)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # logging for eval
    if iter_num % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Iter: {iter_num}, Time: {time.time() - start_time} secs, num tokens {batch_size * block_size * iter_num}, train_loss: {losses['train']} val_loss: {losses['val']}")
        wandb.log({
            'iter' : iter_num,
            'val' : losses['val'],
            'best_val' : best_loss,
            'num_parameters' : model.n_params,
            'train' : losses['train'], 
            'num_tokens' : batch_size * block_size * iter_num,
            'time' : time.time() - start_time,
        })
        # save best checkpoint
        if losses['val'] < best_loss:
            best_loss = losses['val']
            torch.save(model, os.path.join(out_dir, f"{run_name}-gpt-{iter_num}.ckpt"))

    iter_num += 1            
    if iter_num >= max_iters: 
        break
    
