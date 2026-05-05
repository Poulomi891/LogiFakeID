import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
import mmseg.models

ASCFORMER_ROOT = "/Path/to/your/data/Manipulated_ID/RTM/ASCFormer"
sys.path.insert(0, ASCFORMER_ROOT)
CONFIG_PATH    = os.path.join(ASCFORMER_ROOT, 'configs', 'ascformer', 'ascformer_rtm.py')
CKPT_PATH      = os.path.join(ASCFORMER_ROOT, 'ascformer_model.pth')

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_ela(img_bgr, quality=80):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recompressed = cv2.imread(tmp_path)
    finally:
        os.unlink(tmp_path)
    return cv2.absdiff(img_bgr, recompressed).astype(np.float32)

def compute_block_dct(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    padded = np.zeros((new_h, new_w), dtype=np.float32)
    padded[:h, :w] = gray

    dct_img = np.zeros_like(padded)
    for i in range(0, new_h, 8):
        for j in range(0, new_w, 8):
            block = padded[i:i+8, j:j+8]
            dct_img[i:i+8, j:j+8] = cv2.dct(block)
    return dct_img[:h, :w]

def load_model():
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.pretrained = None
    if hasattr(cfg.model.backbone, 'backbone_main'): cfg.model.backbone.backbone_main.pretrained = None
    if hasattr(cfg.model.backbone, 'backbone_extra'): cfg.model.backbone.backbone_extra.pretrained = None
    if 'data_preprocessor' in cfg.model:
        if not cfg.model.data_preprocessor.type.startswith('mmseg.'):
            cfg.model.data_preprocessor.type = 'mmseg.' + cfg.model.data_preprocessor.type
    
    model = MODELS.build(cfg.model)
    model.cfg = cfg
    load_checkpoint(model, CKPT_PATH, map_location='cpu')
    model.to(DEVICE).eval()
    return model, cfg

@torch.no_grad()
def predict_score(model, cfg, img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return 0.0
    h, w = img_bgr.shape[:2]
    ela = compute_ela(img_bgr, quality=80)
    dct = compute_block_dct(img_bgr)

    mean = np.array([123.675, 116.28, 103.53])
    std  = np.array([58.395, 57.12, 57.375])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_norm = (img_rgb - mean) / std

    div = 32
    new_h = ((h + div - 1) // div) * div
    new_w = ((w + div - 1) // div) * div

    img_pad = np.zeros((new_h, new_w, 3), dtype=np.float32)
    img_pad[:h, :w] = img_norm
    img_t = torch.from_numpy(img_pad.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    ela_pad = np.zeros((new_h, new_w, 3), dtype=np.float32)
    ela_pad[:h, :w] = ela
    ela_t = torch.from_numpy(ela_pad.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    dct_pad = np.zeros((new_h, new_w), dtype=np.float32)
    dct_pad[:h, :w] = dct
    dct_t = torch.from_numpy(dct_pad).unsqueeze(0).unsqueeze(0).to(DEVICE)

    extras = {'dct': dct_t, 'ela': ela_t}
    batch_img_metas = [dict(ori_shape=(h, w), img_shape=(new_h, new_w), pad_shape=(new_h, new_w), padding_size=[0, new_w - w, 0, new_h - h], img_path=img_path)]

    try:
        seg_logits = model.whole_inference(img_t, extras, batch_img_metas)
        prob_map = F.softmax(seg_logits, dim=1)[0, 1, :h, :w]
        prob_np = prob_map.float().cpu().numpy()
        flat = prob_np.flatten()
        k = max(1, int(len(flat) * 0.05))
        top_k = np.partition(flat, -k)[-k:]
        return float(top_k.mean())
    except Exception as e:
        return 0.0

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

def test_ascformer_fantasy():
    print(f"\n🚀 Evaluating ASCFormer on Fantasy ID Dataset")
    model, cfg = load_model()

    samples = []
    import pandas as pd
    csv_path = os.path.join(FANTASY_ROOT, "fantasyIDiap-test.csv")
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        rel_path = row['path']
        is_attack = str(row['is_attack']).lower() == 'true'
        label = 1 if is_attack else 0
        abs_path = os.path.join(FANTASY_ROOT, rel_path)
        if os.path.exists(abs_path):
            samples.append((abs_path, label))

    all_scores, all_labels = [], []
    for path, label in tqdm(samples, desc="Scanning"):
        score = predict_score(model, cfg, path)
        all_scores.append(score)
        all_labels.append(label)

    all_scores, all_labels = np.array(all_scores), np.array(all_labels)
    all_preds = (all_scores > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    eer = compute_eer(all_labels, all_scores)
    
    print(f"\n📊 Results: Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")

if __name__ == "__main__":
    test_ascformer_fantasy()
