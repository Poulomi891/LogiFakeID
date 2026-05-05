import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/ProDet"
sys.path.insert(0, REPO_ROOT)
from demo.efficientnetb4 import EfficientNetB4
from test_prodet_pretrained import FeatureAttentionBlock

class ProDetArchi(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = {'mode': 'original', 'num_classes': 2, 'inc': 3, 'dropout': False, 'pretrained': None}
        self.backbone = EfficientNetB4(cfg)
        self.adjust_feature = nn.Conv2d(1792, 512, 1)
        self.fea_att = FeatureAttentionBlock()
        def head(): return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 2))
        self.ID_inconsistency_classifier, self.blend_classifier, self.deepfake_classifier, self.final_classifier = head(), head(), head(), head()
    def forward(self, x):
        f = self.adjust_feature(self.backbone.features(x))
        df, bld, bi = self.deepfake_classifier(f), self.blend_classifier(f), self.ID_inconsistency_classifier(f)
        return self.final_classifier(self.fea_att(df, bld, bi, f))

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
PRETRAINED_WEIGHTS = os.path.join(REPO_ROOT, "ProDet_best.pth")

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
            return torch.zeros(3, 256, 256), label, os.path.basename(path)

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

def test_prodet_fantasy():
    print(f"\n🚀 Evaluating ProDet on Fantasy ID Dataset")
    model = ProDetArchi().to(DEVICE).eval()
    ckpt = torch.load(PRETRAINED_WEIGHTS, weights_only=False, map_location="cpu")
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FantasyDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, names in tqdm(loader, desc="Scanning"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            scores = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            if scores.ndim == 0: scores = np.array([scores])
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    all_scores, all_labels = np.array(all_scores), np.array(all_labels)
    all_preds = (all_scores > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    eer = compute_eer(all_labels, all_scores)
    
    print(f"\n📊 Results: Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")

if __name__ == "__main__":
    test_prodet_fantasy()
