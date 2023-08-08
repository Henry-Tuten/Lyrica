# Import necessary libraries and define your model architecture
import torch
import torch.nn as nn
from GPTLanguageModel import GPTLanguageModel
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm


print("Reading and processing the CSV file")
# Read the CSV file
filename = 'data/processed/first_25000_lines.csv'
data_csv = pd.read_csv(filename)

# Extract the lyrics column
lyrics_list = data_csv['lyrics'].tolist()

# Concatenate all the lyrics together
text = "\n".join(str(lyric) for lyric in lyrics_list)


print("Encoding the text")
# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available(), "CUDA IN GENERATED TEXT")
model_path = "src/models/gpt2_style_model.pth"
print("Model Path added")
model = GPTLanguageModel() # Define the model just as in the training script

# Training is happening upon instantiation

print("Model Instantiated")
model.to(device)
print("Model added to GPU")
# Load pre-trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
# Check to see if training is happening before or after this point
print("Loading Complete")


model.eval()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

#completed training on first set of data