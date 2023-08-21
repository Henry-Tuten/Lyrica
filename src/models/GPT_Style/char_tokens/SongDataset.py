from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

class SongDataset(Dataset):
    def __init__(self, texts, block_size):
        self.texts = texts
        self.block_size = block_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.texts[idx], add_special_tokens=True, truncation=True, max_length=self.block_size)

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)