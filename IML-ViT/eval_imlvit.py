import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
# --- NumPy 2.0 Pickle Compatibility Fix ---
import numpy
import sys
try:
    import numpy._core.numeric as _num
except ImportError:
    pass
if hasattr(numpy, "_core"):
    sys.modules["numpy.core"] = numpy._core
else:
    sys.modules["numpy._core"] = numpy.core
# ------------------------------------------
# --- NumPy 2.0 Pickle Compatibility Fix ---
import numpy
import sys
try:
    import numpy._core.numeric as _num
except ImportError:
    pass
if hasattr(numpy, "_core"):
    sys.modules["numpy.core"] = numpy._core
else:
    sys.modules["numpy._core"] = numpy.core
# ------------------------------------------

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "/Path/to/your/data/Manipulated_ID"
REPO_PATH = os.path.join(ROOT_DIR, "IML-ViT")
CKPT_PATH = os.path.join(REPO_PATH, "checkpoints/IML-ViT_checkpoints/iml-vit_checkpoint.pth")

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

class IMLDataset(Dataset):
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
        path, label = self.samples[idx]; img_pil = Image.open(path).convert("RGB")
        img = self.transform(img_pil); orig_shape = np.array(img_pil.size[::-1])
        return img, label, path, orig_shape

def main():
    sys.path.insert(0, REPO_PATH)
    import iml_vit_model
    print("📦 Loading IML-ViT Model...")
    model = iml_vit_model.iml_vit_model(vit_pretrain_path=None, predict_head_norm="BN", edge_lambda=20).to(DEVICE).eval()
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    if "model" in ckpt: ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=True)

    all_metrics = []
    for mode in ["Template", "Content"]:
        root = TEMPLATE_ROOT if mode == "Template" else CONTENT_ROOT
        if not os.path.exists(root): continue
        dataset = IMLDataset(root); loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        print(f"🚀 Evaluating IML-ViT on {mode} Mode...")
        results = []
        with torch.no_grad():
            for imgs, labels, paths, shapes in tqdm(loader):
                dummy_m, dummy_e = torch.zeros(1, 1, 1024, 1024, device=DEVICE), torch.zeros(1, 1, 1024, 1024, device=DEVICE)
                _, mask_pred, _ = model(imgs.to(DEVICE), dummy_m, dummy_e, shapes.to(DEVICE))
                h, w = shapes[0]; score = mask_pred[0, 0, :h, :w].mean().item() if h > 0 else mask_pred[0, 0].mean().item()
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

    report_df = pd.DataFrame(all_metrics); print("\nIML-VIT MODEL RESULTS\n" + report_df.to_string(index=False)); report_df.to_csv("results_imlvit_per_manipulation.csv", index=False)

if __name__ == "__main__":
    main()
