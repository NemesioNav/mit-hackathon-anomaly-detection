import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import numpy as np
import faiss
from PIL import Image
import os
import cv2
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.hub

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Pretrained model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval().to(DEVICE)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # grayscale to [-1, 1]
    transforms.Lambda(lambda x: x.expand(3, -1, -1))  # convert 1→3 channels
])

def extract_dino_patches(dataloader):
    features = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(DEVICE)
            feats = model.forward_features(x)
            tokens = feats['x_norm_patchtokens']
            tokens = tokens.cpu()
            features.append(tokens)
    return torch.cat(features, dim=0).numpy()

class AugmentedImageFolder(Dataset):
    def __init__(self, root, transform):
        self.dataset = ImageFolder(root, transform=None)  # defer transform
        self.transform = transform
        self.hflip = transforms.RandomHorizontalFlip(p=1.0)

    def __len__(self):
        return 2 * len(self.dataset)

    def __getitem__(self, idx):
        original_idx = idx % len(self.dataset)
        image, label = self.dataset[original_idx]

        if idx < len(self.dataset):
            image = self.transform(image)  # Original
        else:
            image = self.hflip(image)
            image = self.transform(image)  # Flipped

        return image, label

def score_image(img_path, index, pca=None):
    x = transform(Image.open(img_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.forward_features(x)
        tokens = feat['x_norm_patchtokens']
        patches = tokens[0].cpu().numpy()
    # pca
    if pca is not None:
        patches = pca.transform(patches)
    # faiss
    # D, _ = index.search(patches, k=1)
    # score = np.max(D)
    D, _ = index.search(patches, k=3)
    score = np.max(D.mean(axis=1))

    return score

def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Script")
    parser.add_argument('--train_dir', type=str, default="data/train", help="Path to training data directory")
    parser.add_argument('--test_dir', type=str, default="data/test", help="Path to testing data directory")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--coreset_fraction', type=float, default=1.0, help="Fraction of features to keep in coreset")
    parser.add_argument('--pca', action='store_true', help="Use PCA for dimensionality reduction")
    args = parser.parse_args()

    # Load training data
    train_dataset = AugmentedImageFolder(args.train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Extract features
    feature_bank = extract_dino_patches(train_loader)
    feature_bank = np.vstack(feature_bank)

    # Optional: sample coreset
    np.random.seed(0)
    np.random.shuffle(feature_bank)
    coreset = feature_bank[:int(len(feature_bank) * args.coreset_fraction)]

    pca = None
    if args.pca:
        # ------------------------------
        pca = PCA(n_components=150)  # tune between 50–150
        coreset = pca.fit_transform(coreset)

    # Build Faiss index
    index = faiss.IndexFlatL2(coreset.shape[1])
    index.add(coreset)

    # Run test
    test_paths = []
    labels = []  # 0: normal, 1: abnormal
    for label, subdir in enumerate(['normal', 'abnormal']):
        for img in os.listdir(f'{args.test_dir}/{subdir}'):
            test_paths.append(f'{args.test_dir}/{subdir}/{img}')
            labels.append(label)

    scores = [score_image(p, index, pca) for p in test_paths]

    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Compute AUROC
    print("AUROC:", roc_auc_score(labels, scores))

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(labels, scores))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # Save the plot as a PNG file
    plt.savefig('roc_curve.png')  # Save the plot to a file
    plt.close()  # Close the plot to free memory

if __name__ == "__main__":
    main()
