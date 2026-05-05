import os
import sys
import numpy as np
import torch

# ── GPU Stability Patch ──────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
# ─────────────────────────────────────────────────────────────────────────────

import torch.nn.functional as F
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS

# ── Registration of custom components ────────────────────────────────────────
import mmseg.models 
from mmseg.models.data_preprocessor import SegDataPreProcessorWithExtra
from mmseg.models.segmentors.my_model_full import MyModelFull

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ASCFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ASCFORMER_ROOT, 'configs', 'ascformer', 'ascformer_rtm.py')
CKPT_PATH = os.path.join(ASCFORMER_ROOT, 'ascformer_model.pth')

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

def compute_ela(img_bgr, quality=80):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recompressed = cv2.imread(tmp.name)
    ela = cv2.absdiff(img_bgr, recompressed).astype(np.float32)
    os.unlink(tmp.name)
    return ela

def compute_block_dct(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    padded = np.zeros((((h+7)//8)*8, ((w+7)//8)*8), dtype=np.float32)
    padded[:h, :w] = gray
    dct_img = np.zeros_like(padded)
    for i in range(0, padded.shape[0], 8):
        for j in range(0, padded.shape[1], 8):
            dct_img[i:i+8, j:j+8] = cv2.dct(padded[i:i+8, j:j+8])
    return dct_img[:h, :w]

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ASCFormer Evaluation (GPU Stability Mode)")
    print("=" * 60)

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.pretrained = None
    if 'data_preprocessor' in cfg.model:
        if not cfg.model.data_preprocessor.type.startswith('mmseg.'):
            cfg.model.data_preprocessor.type = 'mmseg.' + cfg.model.data_preprocessor.type

    model = MODELS.build(cfg.model)
    load_checkpoint(model, CKPT_PATH, map_location='cpu')
    model.to(DEVICE).eval()

    all_metrics = []
    for mode, root in [("Template", TEMPLATE_ROOT), ("Content", CONTENT_ROOT)]:
        if not os.path.exists(root): continue
        samples = []
        for lbl_name, lbl_val in [("real", 0), ("fake", 1)]:
            folder = os.path.join(root, lbl_name)
            if os.path.exists(folder):
                samples += [(os.path.join(folder, f), lbl_val) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        
        print(f"🚀 Evaluating ASCFormer on {mode} Mode...")
        results = []
        for path, label in tqdm(samples):
            img_bgr = cv2.imread(path)
            if img_bgr is None: continue
            h, w = img_bgr.shape[:2]

            ela = compute_ela(img_bgr)
            dct = compute_block_dct(img_bgr)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_norm = (img_rgb - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])

            div = 32
            nh, nw = ((h+div-1)//div)*div, ((w+div-1)//div)*div
            img_p = np.zeros((nh, nw, 3), dtype=np.float32); img_p[:h, :w] = img_norm
            ela_p = np.zeros((nh, nw, 3), dtype=np.float32); ela_p[:h, :w] = ela
            dct_p = np.zeros((nh, nw), dtype=np.float32); dct_p[:h, :w] = dct

            img_t = torch.from_numpy(img_p.transpose(2,0,1)).unsqueeze(0).to(DEVICE).contiguous()
            ela_t = torch.from_numpy(ela_p.transpose(2,0,1)).unsqueeze(0).to(DEVICE).contiguous()
            dct_t = torch.from_numpy(dct_p).unsqueeze(0).unsqueeze(0).to(DEVICE).contiguous()

            with torch.no_grad():
                logits = model.whole_inference(img_t, {'dct': dct_t, 'ela': ela_t}, [dict(ori_shape=(h, w), img_shape=(nh, nw), pad_shape=(nh, nw), padding_size=[0, nw-w, 0, nh-h], img_path=path)])
                prob_map = F.softmax(logits, dim=1)[0, 1, :h, :w].cpu().numpy().flatten()
                k = max(1, int(len(prob_map) * 0.05))
                score = float(np.partition(prob_map, -k)[-k:].mean())
            
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
    final_df.to_csv("results_ascformer_per_manipulation.csv", index=False)

if __name__ == "__main__":
    main()
