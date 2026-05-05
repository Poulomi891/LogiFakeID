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
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "/Path/to/your/data/Manipulated_ID"
REPO_PATH = os.path.join(ROOT_DIR, "SAFIRE")
CKPT_PATH = os.path.join(REPO_PATH, "SAFIRE-20260421T172202Z-3-001/SAFIRE/safire.pth")

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

class SafireDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label_name in ["real", "fake"]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder): continue
            label = 1 if label_name == "fake" else 0
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')): self.samples.append((os.path.join(folder, f), label))
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]; img = self.transform(Image.open(path).convert("RGB"))
        return img, label, path

def main():
    sys.path.insert(0, REPO_PATH)
    sys.path.insert(0, os.path.join(REPO_PATH, "networks"))
    from segment_anything import sam_model_registry
    from networks.safire_model import AdaptorSAM
    from networks.safire_predictor_binary import SafirePredictor
    
    print(f"📦 Loading SAFIRE Model on {DEVICE}...")
    sam_ckpt = os.path.join(os.path.dirname(CKPT_PATH), "sam_vit_b_01ec64.pth")
    sam_model = sam_model_registry["vit_b_adaptor"](checkpoint=sam_ckpt)
    safire_inner = AdaptorSAM(image_encoder=sam_model.image_encoder, mask_decoder=sam_model.mask_decoder, prompt_encoder=sam_model.prompt_encoder)
    
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location='cpu')
    safire_inner.load_state_dict({k.replace("module.", ""): ckpt["model"][k] for k in ckpt["model"]})
    
    # Move model to device
    safire_inner = safire_inner.to(DEVICE).eval()
    
    predictor = SafirePredictor(safire_inner, points_per_side=16, points_per_batch=64, pred_iou_thresh=0, stability_score_thresh=0, box_nms_thresh=0)


    all_metrics = []
    for mode in ["Template", "Content"]:
        root = TEMPLATE_ROOT if mode == "Template" else CONTENT_ROOT
        if not os.path.exists(root): continue
        dataset = SafireDataset(root); loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        print(f"🚀 Evaluating SAFIRE on {mode} Mode...")
        results = []
        with torch.no_grad():
            for imgs, labels, paths in tqdm(loader):
                img_np = (imgs[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                _, mask_pred, _ = predictor.safire_predict(img_np)
                flat = mask_pred.flatten(); k = max(1, int(len(flat) * 0.05))
                score = float(np.partition(flat, -k)[-k:].mean())
                results.append({"path": paths[0], "score": score, "label": labels[0].item(), "type": get_manipulation_type(paths[0])})
        
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

    report_df = pd.DataFrame(all_metrics); print("\nSAFIRE MODEL RESULTS\n" + report_df.to_string(index=False)); report_df.to_csv("results_safire_per_manipulation.csv", index=False)

if __name__ == "__main__":
    main()
