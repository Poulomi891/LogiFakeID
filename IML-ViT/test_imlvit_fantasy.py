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

# --- NumPy 2.0 Pickle Compatibility Fix ---
import numpy
try:
    import numpy._core.numeric as _num
except ImportError:
    pass
if hasattr(numpy, "_core"):
    sys.modules["numpy.core"] = numpy._core
else:
    sys.modules["numpy._core"] = numpy.core
# ------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/IML-ViT"
sys.path.insert(0, REPO_ROOT)

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints/IML-ViT_checkpoints")
CHECKPOINTS = {
    "Default": "iml-vit_checkpoint.pth",
    "CASIAv2": "iml-vit_checkpoint_casiav2_20231014.pth",
    "TruFor": "iml-vit_checkpoint_trufor_20231104.pth"
}

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
            img_pil = Image.open(path).convert("RGB")
            orig_shape = np.array(img_pil.size[::-1])
            if self.transform: img = self.transform(img_pil)
            return img, label, os.path.basename(path), orig_shape
        except:
            return torch.zeros(3, 1024, 1024), label, os.path.basename(path), np.array([1024, 1024])

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

def test_imlvit_fantasy(ckpt_name, ckpt_file):
    print(f"\n🚀 Evaluating IML-ViT ({ckpt_name}) on Fantasy ID Dataset")
    import iml_vit_model
    
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    model = iml_vit_model.iml_vit_model(vit_pretrain_path=None, predict_head_norm="BN", edge_lambda=20).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if "model" in ckpt: ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=True)

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FantasyDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, names, shapes in tqdm(loader, desc="Scanning"):
            imgs = imgs.to(DEVICE)
            dummy_m = torch.zeros(1, 1, 1024, 1024, device=DEVICE)
            dummy_e = torch.zeros(1, 1, 1024, 1024, device=DEVICE)
            
            _, mask_pred, _ = model(imgs, dummy_m, dummy_e, shapes.to(DEVICE))
            h, w = shapes[0]
            if h > 0 and w > 0:
                score = mask_pred[0, 0, :h, :w].mean().item()
            else:
                score = mask_pred[0, 0].mean().item()
            
            all_scores.append(score)
            all_labels.append(labels[0].item())

    all_scores, all_labels = np.array(all_scores), np.array(all_labels)
    all_preds = (all_scores > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    eer = compute_eer(all_labels, all_scores)
    
    print(f"\n📊 Results ({ckpt_name}): Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for ckpt_name, ckpt_file in CHECKPOINTS.items():
        test_imlvit_fantasy(ckpt_name, ckpt_file)
