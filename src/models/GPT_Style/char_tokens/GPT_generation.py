import torch
import pandas as pd
from GPTLanguageModel import GPTLanguageModel

#pass arguments into gptmodel upon instantiation
batch_size =  64 #64  how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
eval_iters = 200
# dimensionality of the embeddings
n_embd = 384
n_head = 7
n_layer = 7
dropout = 0.2


print("Reading and processing the CSV file")
# Read the CSV file
filename = 'data/processed/eighth_1.csv'
data_csv = pd.read_csv(filename)

# Extract the lyrics column
lyrics_list = data_csv['combined'].tolist()

# Concatenate all the lyrics together
text = "\n".join(str(lyric) for lyric in lyrics_list)


print("Encoding the text")

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)

# Load the model
model.load_state_dict(torch.load('src/models/GPT_style/char_tokens/gpt2_style_master.pth'))
model.eval()

# Move the model to the same device as the context
model = model.to(device)

while True:
    # Ask the user for input
    input_sentence = input("Enter a sentence (or 'exit' to end): ")
    
    if input_sentence.lower() == 'exit':
        print("Exiting the program.")
        break
    
    # Encode the input sentence using the same mapping as the rest of your data
    context = torch.tensor([encode(input_sentence)], dtype=torch.long, device=device)
    
    # Generate the text
    generated_tokens = model.generate(context, block_size, max_new_tokens=500)[0].tolist()
    
    # Decode the generated tokens back into text
    generated_text = decode(generated_tokens)
    
    print("Generated text:")
    print(generated_text)