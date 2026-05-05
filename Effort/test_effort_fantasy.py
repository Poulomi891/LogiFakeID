import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import yaml
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ── Setup ────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/Effort-AIGI-Detection"
BENCH_ROOT = os.path.join(REPO_ROOT, "DeepfakeBench")
sys.path.insert(0, BENCH_ROOT)
sys.path.insert(0, os.path.join(BENCH_ROOT, "training"))

from training.detectors import DETECTOR

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
PRETRAINED_WEIGHTS = os.path.join(REPO_ROOT, "effort_clip_L14_trainOn_FaceForensic.pth")
CONFIG_PATH = os.path.join(BENCH_ROOT, "training/config/detector/effort.yaml")

class FantasyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        import pandas as pd
        csv_path = os.path.join(FANTASY_ROOT, "fantasyIDiap-test.csv")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rel_path = row['path']
            is_attack = str(row['is_attack']).lower() == 'true'
            label = 1 if is_attack else 0
            abs_path = os.path.join(FANTASY_ROOT, rel_path)
            if os.path.exists(abs_path):
                self.samples.append((abs_path, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, label, os.path.basename(path)
        except:
            return torch.zeros(3, 224, 224), label, os.path.basename(path)

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

def test_effort_fantasy():
    print(f"\n🚀 Evaluating EFFORT on Fantasy ID Dataset")
    
    with open(CONFIG_PATH, "r") as f: config = yaml.safe_load(f)
    model = DETECTOR[config.get("detector_type", "effort")](config).to(DEVICE)
    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    dataset = FantasyDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, names in tqdm(loader, desc="Scanning"):
            imgs = imgs.to(DEVICE)
            output = model({"image": imgs}, inference=True)
            logits = output["cls"]
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            if probs.ndim == 0: probs = np.array([probs])
            all_scores.extend(probs)
            all_labels.extend(labels.numpy())

    all_scores, all_labels = np.array(all_scores), np.array(all_labels)
    all_preds = (all_scores > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    eer = compute_eer(all_labels, all_scores)
    
    print(f"\n📊 Results: Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")

if __name__ == "__main__":
    test_effort_fantasy()
