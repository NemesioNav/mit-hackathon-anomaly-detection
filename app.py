# app.py

import streamlit as st
import torch
import numpy as np
from PIL import Image
import faiss
from torchvision import transforms
import torch.hub
import tempfile

# Load model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval().to(DEVICE)

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(lambda x: x.expand(3, -1, -1))
])

# Load FAISS index
index = faiss.read_index("dino_faiss.index")

# Scoring function
def score_image_patches(image: Image.Image):
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = model.forward_features(x)
        patches = feats['x_norm_patchtokens'][0].cpu().numpy()
    D, _ = index.search(patches, k=3)
    return np.max(D.mean(axis=1))

# Streamlit UI
st.set_page_config(page_title="Thermal Anomaly Detection", layout="centered")
st.title("🚁 Thermal Drone Anomaly Detection")
st.write("Upload a thermal drone image to check for potential human presence (anomaly).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
threshold = st.slider("Anomaly Score Threshold", 0.0, 2000.0, 1296.9338, 0.1)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        score = score_image_patches(image)
        st.write(f"**Anomaly Score:** {score:.4f}")
        if score > threshold:
            st.error("🚨 Anomaly Detected: Potential human presence")
        else:
            st.success("✅ Normal: No anomaly detected")