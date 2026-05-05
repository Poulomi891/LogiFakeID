import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "/Path/to/your/data/Manipulated_ID"
REPO_PATH = os.path.join(ROOT_DIR, "Effort-AIGI-Detection")
CKPT_PATH = os.path.join(ROOT_DIR, "Effort-AIGI-Detection/effort_clip_L14_trainOn_FaceForensic.pth")

# Datasets
TEMPLATE_ROOT = os.path.join(ROOT_DIR, "Final_ID_dataset_split_template/test")
CONTENT_ROOT = os.path.join(ROOT_DIR, "Final_ID_dataset_split_content/test")

# ── Manipulation Mapping ─────────────────────────────────────────────────────
GROUPS = {
    "Template": {
        "Spelling Errors": ["doc_spelling", "gov_spelling"],
        "Word Jumbling": ["authority"],
        "General Info Substitution": ["email_change", "phone_change", "website_change"]
    },
    "Content": {
        "Invalid Dates": ["invalid_issue_date", "invalid_details_date", "invalid_dob"],
        "Invalid Aadhar Number": ["invalid_id"],
        "Gender Mismatch": ["gender_mismatch"]
    }
}

def get_manipulation_type(path):
    filename = os.path.basename(path).lower()
    for category in ["authority", "doc_spelling", "gov_spelling", "email_change", "phone_change", "website_change", "gender_mismatch", "invalid_issue_date", "invalid_details_date", "invalid_dob", "invalid_id"]:
        if category in filename: return category
    return "unknown"

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except:
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

# ── Dataset ──────────────────────────────────────────────────────────────────
class ForensicDataset(Dataset):
    def __init__(self, root_dir, size=(224, 224)):
        self.samples = []
        for label_name in ["real", "fake"]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder): continue
            label = 1 if label_name == "fake" else 0
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(folder, f), label))
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, label, path

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    sys.path.insert(0, REPO_PATH)
    sys.path.insert(0, os.path.join(REPO_PATH, "DeepfakeBench"))
    from training.detectors import DETECTOR
    import yaml
    
    with open(os.path.join(REPO_PATH, "DeepfakeBench/training/config/detector/effort.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    
    print("📦 Loading Effort Model...")
    model = DETECTOR["effort"](cfg).to(DEVICE).eval()
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)

    all_metrics = []
    
    for mode in ["Template", "Content"]:
        root = TEMPLATE_ROOT if mode == "Template" else CONTENT_ROOT
        if not os.path.exists(root): continue
        
        dataset = ForensicDataset(root)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
        
        print(f"🚀 Evaluating Effort on {mode} Mode...")
        results = []
        with torch.no_grad():
            for imgs, labels, paths in tqdm(loader):
                out = model({"image": imgs.to(DEVICE)}, inference=True)
                scores = torch.softmax(out["cls"], dim=1)[:, 1].cpu().numpy()
                for p, s, l in zip(paths, scores, labels.numpy()):
                    results.append({"path": p, "score": s, "label": l, "type": get_manipulation_type(p)})
        
        df = pd.DataFrame(results)
        real_scores = df[df["label"] == 0]["score"].values
        
        # Overall
        if len(real_scores) > 0 and len(df[df["label"] == 1]) > 0:
            auc = roc_auc_score(df["label"], df["score"])
            eer = compute_eer(df["label"].values, df["score"].values)
            all_metrics.append({"Mode": mode, "Category": "OVERALL", "AUC": auc, "EER": eer})

        # Per Category
        for group_name, manip_list in GROUPS[mode].items():
            m_df = df[(df["label"] == 1) & (df["type"].isin(manip_list))]
            if len(m_df) > 0 and len(real_scores) > 0:
                sub_scores = np.concatenate([real_scores, m_df["score"].values])
                sub_labels = [0]*len(real_scores) + [1]*len(m_df)
                auc, eer = roc_auc_score(sub_labels, sub_scores), compute_eer(sub_labels, sub_scores)
                all_metrics.append({"Mode": mode, "Category": group_name, "AUC": auc, "EER": eer})

    report_df = pd.DataFrame(all_metrics)
    print("\n" + "="*80 + "\nEFFORT MODEL RESULTS\n" + "="*80)
    print(report_df.to_string(index=False))
    report_df.to_csv("results_effort_per_manipulation.csv", index=False)

if __name__ == "__main__":
    main()
