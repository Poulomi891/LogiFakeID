import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "/Path/to/your/data/Manipulated_ID"
REPO_PATH = os.path.join(ROOT_DIR, "ProDet")
CKPT_PATH = os.path.join(REPO_PATH, "ProDet_best.pth")

TEMPLATE_ROOT = os.path.join(ROOT_DIR, "Final_ID_dataset_split_template/test")
CONTENT_ROOT = os.path.join(ROOT_DIR, "Final_ID_dataset_split_content/test")

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
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

class ProDetDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label_name in ["real", "fake"]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder): continue
            label = 1 if label_name == "fake" else 0
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')): self.samples.append((os.path.join(folder, f), label))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]; img = self.transform(Image.open(path).convert("RGB"))
        return img, label, path

def main():
    sys.path.insert(0, REPO_PATH)
    from demo.efficientnetb4 import EfficientNetB4
    from test_prodet_pretrained import FeatureAttentionBlock
    class ProDetArchi(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = {'mode': 'original', 'num_classes': 2, 'inc': 3, 'dropout': False, 'pretrained': None}
            self.backbone = EfficientNetB4(cfg); self.adjust_feature = nn.Conv2d(1792, 512, 1); self.fea_att = FeatureAttentionBlock()
            def head(): return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 2))
            self.ID_inconsistency_classifier, self.blend_classifier, self.deepfake_classifier, self.final_classifier = head(), head(), head(), head()
        def forward(self, x):
            f = self.adjust_feature(self.backbone.features(x)); df, bld, bi = self.deepfake_classifier(f), self.blend_classifier(f), self.ID_inconsistency_classifier(f)
            return self.final_classifier(self.fea_att(df, bld, bi, f))
    
    print("📦 Loading ProDet Model...")
    model = ProDetArchi().to(DEVICE).eval()
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)

    all_metrics = []
    for mode in ["Template", "Content"]:
        root = TEMPLATE_ROOT if mode == "Template" else CONTENT_ROOT
        if not os.path.exists(root): continue
        dataset = ProDetDataset(root); loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
        print(f"🚀 Evaluating ProDet on {mode} Mode...")
        results = []
        with torch.no_grad():
            for imgs, labels, paths in tqdm(loader):
                logits = model(imgs.to(DEVICE)); scores = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                for p, s, l in zip(paths, scores, labels.numpy()): results.append({"path": p, "score": s, "label": l, "type": get_manipulation_type(p)})
        df = pd.DataFrame(results); real_scores = df[df["label"] == 0]["score"].values
        
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

    report_df = pd.DataFrame(all_metrics); print("\nPRODET MODEL RESULTS\n" + report_df.to_string(index=False)); report_df.to_csv("results_prodet_per_manipulation.csv", index=False)

if __name__ == "__main__":
    main()
