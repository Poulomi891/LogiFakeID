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
REPO_PATH = os.path.join(ROOT_DIR, "MMFusion-IML")
CKPT_PATH = os.path.join(REPO_PATH, "ckpt/early_fusion_detection.pth")

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

class MMDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label_name in ["real", "fake"]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder): continue
            label = 1 if label_name == "fake" else 0
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')): self.samples.append((os.path.join(folder, f), label))
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]; img = self.transform(Image.open(path).convert("RGB"))
        return img, label, path

def main():
    sys.path.insert(0, REPO_PATH)
    from models.cmnext_conf import CMNeXtWithConf
    from models.modal_extract import ModalitiesExtractor
    from configs.cmnext_init_cfg import _C as mm_cfg
    mm_cfg.defrost(); mm_cfg.MODEL.MODALS, mm_cfg.MODEL.BACKBONE, mm_cfg.MODEL.DETECTION, mm_cfg.MODEL.TRAIN_PHASE = ['img', 'noiseprint', 'bayar', 'srm'], 'MixCMNeXtMHSA-B2', 'confpool', 'detection'; mm_cfg.freeze()
    extractor = ModalitiesExtractor(mm_cfg.MODEL.MODALS[1:], None); backbone = CMNeXtWithConf(mm_cfg.MODEL)
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    backbone.load_state_dict(ckpt['state_dict']); extractor.load_state_dict(ckpt['extractor_state_dict'])
    
    class MMFusionWrapper(nn.Module):
        def __init__(self, ext, bb): super().__init__(); self.ext, self.bb = ext, bb
        def forward(self, x):
            modals = self.ext(x); mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1); std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1); x_norm = (x - mean) / std
            _, _, det = self.bb([x_norm] + modals); return det
    
    print("📦 Loading MMFusion Model...")
    model = MMFusionWrapper(extractor, backbone).to(DEVICE).eval()

    all_metrics = []
    for mode in ["Template", "Content"]:
        root = TEMPLATE_ROOT if mode == "Template" else CONTENT_ROOT
        if not os.path.exists(root): continue
        dataset = MMDataset(root); loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        print(f"🚀 Evaluating MMFusion on {mode} Mode...")
        results = []
        with torch.no_grad():
            for imgs, labels, paths in tqdm(loader):
                det = model(imgs.to(DEVICE)); scores = torch.sigmoid(det).cpu().numpy().flatten()
                for p, s, l in zip(paths, scores, labels.numpy()): results.append({"path": p, "score": s, "label": l, "type": get_manipulation_type(p)})
        df = pd.DataFrame(results); real_scores = df[df["label"] == 0]["score"].values
        
        # Overall
        if len(real_scores) > 0 and len(df[df["label"] == 1]) > 0:
            auc, eer = roc_auc_score(df["label"], df["score"]), compute_eer(df["label"].values, df["score"].values)
            all_metrics.append({"Mode": mode, "Category": "OVERALL", "AUC": auc, "EER": eer})
            
        # Per Category
        for group_name, manip_list in GROUPS[mode].items():
            m_df = df[(df["label"] == 1) & (df["type"].isin(manip_list))]
            if len(m_df) > 0 and len(real_scores) > 0:
                sub_scores = np.concatenate([real_scores, m_df["score"].values])
                sub_labels = [0]*len(real_scores) + [1]*len(m_df)
                auc, eer = roc_auc_score(sub_labels, sub_scores), compute_eer(sub_labels, sub_scores)
                all_metrics.append({"Mode": mode, "Category": group_name, "AUC": auc, "EER": eer})

    report_df = pd.DataFrame(all_metrics); print("\nMMFUSION MODEL RESULTS\n" + report_df.to_string(index=False)); report_df.to_csv("results_mmfusion_per_manipulation.csv", index=False)

if __name__ == "__main__":
    main()
