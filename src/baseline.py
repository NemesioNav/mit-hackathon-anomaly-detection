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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-2])  # Remove classifier
model.eval().to(DEVICE)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Thermal to RGB
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet stats
                         [0.229, 0.224, 0.225])
])

def extract_features(dataloader):
    features = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(DEVICE)
            feat = model(x)
            feat = feat.view(feat.size(0), feat.size(1), -1)  # B, C, H*W
            feat = feat.permute(0, 2, 1)  # B, N_patches, C
            features.append(feat.cpu())
    return torch.cat(features, dim=0).reshape(-1, 512).numpy()  # All patches

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
        feat = model(x)
        feat = feat.view(feat.size(0), feat.size(1), -1).permute(0, 2, 1)
        patches = feat[0].cpu().numpy()
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
    args = parser.parse_args()

    # Load training data
    train_dataset = AugmentedImageFolder(args.train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Extract features
    feature_bank = extract_features(train_loader)
    print("Feature bank shape:", feature_bank.shape)

    # Optional: sample coreset
    np.random.shuffle(feature_bank)
    coreset = feature_bank[:int(len(feature_bank) * args.coreset_fraction)]

    # ------------------------------
    pca = PCA(n_components=100)  # tune between 50–150
    train_pca = pca.fit_transform(coreset)

    index = faiss.IndexFlatL2(train_pca.shape[1])
    index.add(train_pca)
    # ------------------------------

    # # Build Faiss index
    # index = faiss.IndexFlatL2(coreset.shape[1])
    # index.add(coreset)

    # Run test
    test_paths = []
    labels = []  # 0: normal, 1: abnormal
    for label, subdir in enumerate(['normal', 'abnormal']):
        for img in os.listdir(f'{args.test_dir}/{subdir}'):
            test_paths.append(f'{args.test_dir}/{subdir}/{img}')
            labels.append(label)

    scores = [score_image(p, index, pca) for p in test_paths]

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
    plt.show()

if __name__ == "__main__":
    main()
