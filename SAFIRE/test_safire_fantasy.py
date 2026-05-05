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

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/SAFIRE"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "networks"))

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
CKPT_PATH = os.path.join(REPO_ROOT, "SAFIRE-20260421T172202Z-3-001/SAFIRE/safire.pth")

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
            return torch.zeros(3, 1024, 1024), label, os.path.basename(path)

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

def test_safire_fantasy():
    print(f"\n🚀 Evaluating SAFIRE on Fantasy ID Dataset")
    from segment_anything import sam_model_registry
    from networks.safire_model import AdaptorSAM
    from networks.safire_predictor_binary import SafirePredictor
    
    sam_ckpt = os.path.join(os.path.dirname(CKPT_PATH), "sam_vit_b_01ec64.pth")
    sam_model = sam_model_registry["vit_b_adaptor"](checkpoint=sam_ckpt)
    safire_inner = AdaptorSAM(image_encoder=sam_model.image_encoder, mask_decoder=sam_model.mask_decoder, prompt_encoder=sam_model.prompt_encoder)
    
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location='cpu')
    safire_inner.load_state_dict({k.replace("module.", ""): ckpt["model"][k] for k in ckpt["model"]})
    
    safire_inner = safire_inner.to(DEVICE).eval()
    predictor = SafirePredictor(safire_inner, points_per_side=16, points_per_batch=64, pred_iou_thresh=0, stability_score_thresh=0, box_nms_thresh=0)

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FantasyDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, names in tqdm(loader, desc="Scanning"):
            img_np = (imgs[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            _, mask_pred, _ = predictor.safire_predict(img_np)
            flat = mask_pred.flatten()
            k = max(1, int(len(flat) * 0.05))
            score = float(np.partition(flat, -k)[-k:].mean())
            all_scores.append(score)
            all_labels.append(labels[0].item())

    all_scores, all_labels = np.array(all_scores), np.array(all_labels)
    all_preds = (all_scores > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    eer = compute_eer(all_labels, all_scores)
    
    print(f"\n📊 Results: Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")

if __name__ == "__main__":
    test_safire_fantasy()
