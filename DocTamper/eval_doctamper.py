import os
import sys
import cv2
import torch
import numpy as np
import pickle
import tempfile
import jpegio
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ── NumPy 2.0 Compatibility Unpickler ────────────────────────────────────────
class NumPyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "numpy.core" in module:
            module = module.replace("numpy.core", "numpy._core")
        try:
            return super().find_class(module, name)
        except ImportError:
            if "numpy._core" in module:
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

def load_qt_safe(qt_path):
    with open(qt_path, 'rb') as f:
        pks_ = NumPyUnpickler(f).load()
    return {k: torch.LongTensor(v) for k, v in pks_.items()}

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DT_ROOT = "/Path/to/your/data/Manipulated_ID/DocTamper"
CHECKPOINT = os.path.join(DT_ROOT, "checkpoints/dtd_doctamper.pth")
QT_PATH = os.path.join(DT_ROOT, 'qt_table.pk')

TEMPLATE_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template/test"
CONTENT_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content/test"

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

# ── Import Model Classes into __main__ ───────────────────────────────────────
sys.path.insert(0, os.path.join(DT_ROOT, "models"))
from swins import BasicLayer, SwinTransformerBlock, WindowAttention, Mlp, PatchMerging, PatchEmbed, SwinTransformerV2
from dtd import seg_dtd, DTD, VPH, AddCoords, ConvBlock, LayerNorm, SCSEModule, ConvBNReLU, FUSE1, FUSE2, FUSE3, MID

import __main__
__main__.BasicLayer = BasicLayer
__main__.SwinTransformerBlock = SwinTransformerBlock
__main__.WindowAttention = WindowAttention
__main__.Mlp = Mlp
__main__.PatchMerging = PatchMerging
__main__.PatchEmbed = PatchEmbed
__main__.SwinTransformerV2 = SwinTransformerV2

# ── Utility Functions ────────────────────────────────────────────────────────
def get_manipulation_type(path):
    fn = os.path.basename(path).lower()
    for cat in ["authority", "doc_spelling", "gov_spelling", "email_change", "phone_change", "website_change", "gender_mismatch", "invalid_issue_date", "invalid_details_date", "invalid_dob", "invalid_id"]:
        if cat in fn: return cat
    return "unknown"

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    import timm.models.layers
    sys.modules['timm.models.layers.drop'] = timm.models.layers
    
    print("📦 Loading DocTamper & Data...")
    qts = load_qt_safe(QT_PATH)
    
    for f in ['vph_imagenet.pt', 'swin_imagenet.pt']:
        if not os.path.exists(f):
            try: os.symlink(os.path.join(DT_ROOT, 'checkpoints', f), f)
            except: pass

    model = seg_dtd('', 2).to(DEVICE).eval()
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)

    # ── Compatibility Fixes ──────────────────────────────────────────────────
    for m in model.modules():
        if isinstance(m, torch.nn.GELU):
            if not hasattr(m, 'approximate'): m.approximate = 'none'
        if 'DropPath' in str(type(m)):
            if not hasattr(m, 'scale_by_keep'): m.scale_by_keep = True
    # ─────────────────────────────────────────────────────────────────────────

    to_tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
    ])

    all_metrics = []
    for mode, root in [("Template", TEMPLATE_ROOT), ("Content", CONTENT_ROOT)]:
        if not os.path.exists(root): continue
        samples = []
        for lbl_name, lbl_val in [("real", 0), ("fake", 1)]:
            folder = os.path.join(root, lbl_name)
            if os.path.exists(folder):
                samples += [(os.path.join(folder, f), lbl_val) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        
        print(f"🚀 Evaluating DocTamper on {mode}...")
        results = []
        for path, label in tqdm(samples):
            im = Image.open(path).convert('RGB')
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                im.resize((512, 512)).save(tmp.name, "JPEG", quality=95)
                dct = np.clip(np.abs(jpegio.read(tmp.name).coef_arrays[0].copy()), 0, 20)
            
            img_t = to_tensor(im).unsqueeze(0).to(DEVICE)
            dct_t = torch.from_numpy(dct).long().unsqueeze(0).to(DEVICE)
            qtb_t = qts[95].reshape(1, 8, 8).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(img_t, dct_t, qtb_t)
                score = F.softmax(logits, dim=1)[0, 1, :, :].max().item()
            
            results.append({"score": score, "label": label, "type": get_manipulation_type(path)})

        df = pd.DataFrame(results)
        real_scores = df[df["label"] == 0]["score"].values
        if len(real_scores) > 0:
            auc, eer = roc_auc_score(df["label"], df["score"]), compute_eer(df["label"].values, df["score"].values)
            all_metrics.append({"Mode": mode, "Category": "OVERALL", "AUC": auc, "EER": eer})
            for cat_name, m_list in GROUPS[mode].items():
                m_df = df[(df["label"] == 1) & (df["type"].isin(m_list))]
                if len(m_df) > 0:
                    s_labels = [0]*len(real_scores) + [1]*len(m_df)
                    s_scores = np.concatenate([real_scores, m_df["score"].values])
                    all_metrics.append({"Mode": mode, "Category": cat_name, "AUC": roc_auc_score(s_labels, s_scores), "EER": compute_eer(s_labels, s_scores)})

    final_df = pd.DataFrame(all_metrics)
    print("\n" + final_df.to_string(index=False))
    final_df.to_csv("results_doctamper_per_manipulation.csv", index=False)

if __name__ == "__main__":
    main()
