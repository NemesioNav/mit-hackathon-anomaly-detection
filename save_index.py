# save_index.py

import torch
import numpy as np
import faiss
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from dino import model, extract_dino_patches, AugmentedImageFolder, transform  # reuse your definitions

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_faiss_index(train_dir, batch_size=16, coreset_fraction=1.0):
    dataset = AugmentedImageFolder(train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    features = extract_dino_patches(loader)

    feature_bank = np.vstack(features)
    np.random.seed(0)
    np.random.shuffle(feature_bank)
    coreset = feature_bank[:int(len(feature_bank) * coreset_fraction)]

    index = faiss.IndexFlatL2(coreset.shape[1])
    index.add(coreset)

    faiss.write_index(index, "dino_faiss.index")
    print("Saved FAISS index with shape:", coreset.shape)

if __name__ == "__main__":
    save_faiss_index(train_dir="data/train")
