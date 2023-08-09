import torch
import pandas as pd
from transformers import BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# try with tik token

print("Reading and processing the CSV file")
# Read the CSV file
filename = 'data/processed/first_25000_lines.csv'
data_csv = pd.read_csv(filename)

# Extract the lyrics column
lyrics_list = data_csv['lyrics'].tolist()

# Concatenate all the lyrics together
text = "\n".join(str(lyric) for lyric in lyrics_list)

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print("Data has been split into training and validation sets")
# once we have all the tensors, feed them to a transformer model

# send data to standalone file then implement using

# heirarchy in OG repository