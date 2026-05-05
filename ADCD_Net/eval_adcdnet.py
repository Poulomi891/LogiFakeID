import os
import sys
import pickle
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ── NumPy 2.0 Compatibility Unpickler ────────────────────────────────────────
class NumPyUnpickler(pickle.Unpickler):
    """Fixes ModuleNotFoundError: No module named 'numpy._core.numeric'"""
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
ADCD_ROOT = os.path.dirname(os.path.abspath(__file__))
EXP_DATA = os.path.join(ADCD_ROOT, 'ADCD-Net_exp_data')
CKPT_PATH = os.path.join(EXP_DATA, 'ADCDNet.pth')
DOCRES_PATH = os.path.join(EXP_DATA, 'docres.pkl')
QT_PATH = os.path.join(EXP_DATA, 'qt_table.pk')

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

def extract_dct(img_pil):
    from jpeg2dct.numpy import load as dct_load
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_pil.convert("L").save(tmp.name, "JPEG", quality=100)
        dct_y, _, _ = dct_load(tmp.name, normalized=False)
        reopened = Image.open(tmp.name).convert('RGB')
    os.unlink(tmp.name)
    rows, cols, _ = dct_y.shape
    dct = np.empty((8*rows, 8*cols), dtype=np.int32)
    for j in range(rows):
        for i in range(cols): dct[8*j:8*(j+1), 8*i:8*(i+1)] = dct_y[j, i].reshape(8, 8)
    return dct, reopened

def pad_to_16(t):
    h, w = t.shape[-2:]
    nh, nw = ((h+15)//16)*16, ((w+15)//16)*16
    ns = max(nh, nw)
    ph, pw = ns-h, ns-w
    return F.pad(t, (0, pw, 0, ph), value=0), h, w

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    sys.path.insert(0, ADCD_ROOT)
    import cfg; cfg.docres_ckpt_path = DOCRES_PATH
    from model.model import ADCDNet

    print("📦 Loading ADCD-Net & Data...")
    qts = load_qt_safe(QT_PATH)
    model = ADCDNet().to(DEVICE).eval()
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()}, strict=False)

    all_metrics = []
    for mode, root in [("Template", TEMPLATE_ROOT), ("Content", CONTENT_ROOT)]:
        if not os.path.exists(root): continue
        samples = []
        for lbl_name, lbl_val in [("real", 0), ("fake", 1)]:
            folder = os.path.join(root, lbl_name)
            if os.path.exists(folder):
                samples += [(os.path.join(folder, f), lbl_val) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        
        print(f"🚀 Evaluating {mode}...")
        results = []
        for path, label in tqdm(samples):
            img_cv = cv2.imread(path)
            if img_cv is None: continue
            if max(img_cv.shape[:2]) > 512:
                scale = 512 / max(img_cv.shape[:2])
                img_cv = cv2.resize(img_cv, (int(img_cv.shape[1]*scale), int(img_cv.shape[0]*scale)))
            
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            dct, reopened = extract_dct(img_pil)
            
            img_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.455, 0.406), (0.229, 0.224, 0.225))])(reopened)
            img_t, oh, ow = pad_to_16(img_t)
            dct_t, _, _ = pad_to_16(torch.tensor(np.clip(np.abs(dct), 0, 20), dtype=torch.long))
            
            with torch.no_grad():
                img_t, dct_t = img_t.unsqueeze(0).to(DEVICE), dct_t.unsqueeze(0).to(DEVICE)
                qt_t = qts[100].unsqueeze(0).to(DEVICE)
                logits = model(img_t, dct_t, qt_t, torch.zeros_like(img_t[:,:1]), torch.zeros_like(img_t[:,:1]), is_train=False)[0]
                score = F.softmax(logits, dim=1)[0, 1, :oh, :ow].max().item()
            
            results.append({"score": score, "label": label, "type": get_manipulation_type(path)})

        df = pd.DataFrame(results)
        real_scores = df[df["label"] == 0]["score"].values
        if len(real_scores) > 0:
            # Overall
            auc, eer = roc_auc_score(df["label"], df["score"]), compute_eer(df["label"].values, df["score"].values)
            all_metrics.append({"Mode": mode, "Category": "OVERALL", "AUC": auc, "EER": eer})
            # Categories
            for cat_name, m_list in GROUPS[mode].items():
                m_df = df[(df["label"] == 1) & (df["type"].isin(m_list))]
                if len(m_df) > 0:
                    s_labels = [0]*len(real_scores) + [1]*len(m_df)
                    s_scores = np.concatenate([real_scores, m_df["score"].values])
                    all_metrics.append({"Mode": mode, "Category": cat_name, "AUC": roc_auc_score(s_labels, s_scores), "EER": compute_eer(s_labels, s_scores)})

    final_df = pd.DataFrame(all_metrics)
    print("\n" + final_df.to_string(index=False))
    final_df.to_csv("results_adcdnet_per_manipulation.csv", index=False)

if __name__ == "__main__":
    from torchvision import transforms
    main()
