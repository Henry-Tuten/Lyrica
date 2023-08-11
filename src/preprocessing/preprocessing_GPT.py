import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict

print("Reading and processing the CSV file")
# Read the CSV file
filename = 'data/processed/first_25000_lines.csv'
data_csv = pd.read_csv(filename)

# Extract the lyrics column
lyrics_list = data_csv['lyrics'].tolist()

# Concatenate all the lyrics together
text = "\n".join(str(lyric) for lyric in lyrics_list)


# Tokenize the text
tokens = word_tokenize(text)

# Build a vocabulary
vocab = defaultdict(lambda: len(vocab))
vocab['<UNK>'] = 0  # Reserve 0 for unknown or out-of-vocabulary words
encoded_tokens = [vocab[token] for token in tokens]

# Print the vocabulary and encoded tokens
print("Vocabulary:", vocab)
print("Encoded tokens:", encoded_tokens)