import os
import sys
import torch
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

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/TruFor/TruFor_train_test"
sys.path.insert(0, REPO_ROOT)

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
PRETRAINED_WEIGHTS = os.path.join(REPO_ROOT, "pretrained_models/trufor.pth.tar")

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

def test_trufor_fantasy():
    print(f"\n🚀 Evaluating TruFor on Fantasy ID Dataset")
    from lib.config import config as tru_cfg
    from lib.utils import get_model as get_tru_model
    
    tru_cfg.defrost()
    tru_cfg.merge_from_file(os.path.join(REPO_ROOT, 'lib/config/trufor_ph3.yaml'))
    tru_cfg.TEST.MODEL_FILE = PRETRAINED_WEIGHTS
    tru_cfg.freeze()
    
    model = get_tru_model(tru_cfg).to(DEVICE).eval()
    ckpt = torch.load(PRETRAINED_WEIGHTS, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt['state_dict'], strict=True)

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
            _, _, det, _ = model(imgs)
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
    test_trufor_fantasy()
