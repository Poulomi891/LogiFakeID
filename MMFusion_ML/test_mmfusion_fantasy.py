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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/MMFusion-IML"
sys.path.insert(0, REPO_ROOT)

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
PRETRAINED_WEIGHTS = os.path.join(REPO_ROOT, "ckpt/early_fusion_detection.pth")

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
            return torch.zeros(3, 512, 512), label, os.path.basename(path)

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

def test_mmfusion_fantasy():
    print(f"\n🚀 Evaluating MMFusion on Fantasy ID Dataset")
    from models.cmnext_conf import CMNeXtWithConf
    from models.modal_extract import ModalitiesExtractor
    from configs.cmnext_init_cfg import _C as mm_cfg
    
    mm_cfg.defrost()
    mm_cfg.MODEL.MODALS = ['img', 'noiseprint', 'bayar', 'srm']
    mm_cfg.MODEL.BACKBONE = 'MixCMNeXtMHSA-B2'
    mm_cfg.MODEL.DETECTION = 'confpool'
    mm_cfg.MODEL.TRAIN_PHASE = 'detection'
    mm_cfg.freeze()
    
    extractor = ModalitiesExtractor(mm_cfg.MODEL.MODALS[1:], None)
    backbone = CMNeXtWithConf(mm_cfg.MODEL)
    ckpt = torch.load(PRETRAINED_WEIGHTS, weights_only=False, map_location="cpu")
    backbone.load_state_dict(ckpt['state_dict'])
    extractor.load_state_dict(ckpt['extractor_state_dict'])
    
    class MMFusionWrapper(nn.Module):
        def __init__(self, ext, bb): super().__init__(); self.ext, self.bb = ext, bb
        def forward(self, x):
            modals = self.ext(x); mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1); std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1); x_norm = (x - mean) / std
            _, _, det = self.bb([x_norm] + modals); return det
    
    model = MMFusionWrapper(extractor, backbone).to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FantasyDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, names in tqdm(loader, desc="Scanning"):
            imgs = imgs.to(DEVICE)
            det = model(imgs)
            scores = torch.sigmoid(det).cpu().numpy().flatten()
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
    test_mmfusion_fantasy()
