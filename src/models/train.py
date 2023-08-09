import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
from GPTLanguageModel import GPTLanguageModel
from functions import get_batch, estimate_loss

#pass arguments into gptmodel upon instantiation
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 15000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# take the lines from first_25000 lyrics column 

# use them to generate input ids, attention mask, positional encodings

# once we have all the tensors, feed them to a transformer model

print("Loading the model")
model = GPTLanguageModel()

m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

save_path = 'src/models/gpt2_style_model100k.pth'


    
print("Starting the training loop")
for iter in tqdm(range(max_iters), desc="Training Progress"):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    # split, device, train_data, block_size, batch_size, val_data
    xb, yb = get_batch('train', device, )

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the trained model
torch.save(model.state_dict(), save_path)