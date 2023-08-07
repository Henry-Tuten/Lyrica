# Import necessary libraries and define your model architecture
import torch
import torch.nn as nn
from gpt2_style import GPTLanguageModel
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm

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

