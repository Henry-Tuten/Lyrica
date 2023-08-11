import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
from GPTLanguageModel import GPTLanguageModel
from nltk.tokenize import word_tokenize
from collections import defaultdict
import sys


#pass arguments into gptmodel upon instantiation
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
eval_iters = 200
# dimensionality of the embeddings
n_embd = 50 # 384
n_head = 6 # was 6 before
n_layer = 6 # was 6 before
dropout = 0.2


torch.manual_seed(1337)

print("Reading and processing the CSV file")
# Read the CSV file
filename = 'data/processed/first_25000_lines.csv'
data_csv = pd.read_csv(filename)

# Extract the lyrics column
lyrics_list = data_csv['lyrics'].tolist()

print("Encoding the text")

# Build a vocabulary of words
vocab = defaultdict(lambda: len(vocab))
vocab['<UNK>'] = 0  # Reserve 0 for unknown or out-of-vocabulary words

# Tokenize and concatenate all the lyrics together
tokens = [word_tokenize(str(lyric)) for lyric in lyrics_list]
text = [token for lyric in tokens for token in lyric]

# Process the tokens to assign an ID to each unique token in vocab
for token in text:
    vocab[token]

# Build the itos dictionary
itos = {i: ch for ch, i in vocab.items()}

# Determine the vocabulary size
vocab_size = len(vocab)

# Encode the text
encode = lambda s: [vocab[token] for token in s]
decode = lambda l: ' '.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)

# Load the model
model.load_state_dict(torch.load('src/models/gpt2_style_model25k_nltk.pth'))

model.eval()
print("MODE LSET TO EVAL")
# Define your input sentence here
input_sentence = "This is a pop song"

# Encode the input sentence using the same mapping as the rest of your data
context = torch.tensor([encode(input_sentence)], dtype=torch.long, device=device)
print("context established")
# Move the model to the same device as the context
model = model.to(device)
print("model sent to device")
# Generate the text
generated_tokens = model.generate(context, block_size, max_new_tokens=500)[0].tolist()

# Decode the generated tokens back into text
generated_text = decode(generated_tokens)

print(generated_text)