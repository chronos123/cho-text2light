import torch
import clip
from PIL import Image
import numpy as np

device = "cuda:2" if torch.cuda.is_available() else "cpu"

prompts_file = "all_prompts.txt"
with open(prompts_file, 'r') as f:
    prompts = f.read().splitlines()
# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Text prompts
text = clip.tokenize(prompts).to(device)

# Encode text prompts
with torch.no_grad():
    text_features = model.encode_text(text)

# Save embeddings
np.save('clip_emb_ours.npy', text_features.cpu().numpy())
