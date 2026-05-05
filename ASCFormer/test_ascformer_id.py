"""
Evaluate pre-trained ASCFormer on the ID card dataset across four
regional strategies: Full ID, Face, Template, Content.

ASCFormer is a *localization* model for text manipulation detection,
built on MMSegmentation. It uses dual-stream encoding (RGB + DCT/SRM/ELA
forensic features).

For *detection* (image-level real/fake classification), we derive a score
from the predicted localization map:
    score = mean of top-5% pixel probabilities of class 1 (tampered)
    → high score ⇒ model thinks the image is tampered

Metrics reported:
    - Overall Accuracy
    - AUC (Area Under ROC Curve)
    - Per-Class Accuracy  (Real accuracy, Fake accuracy)

Labels:  Real = 0,  Fake = 1  (consistent with your other pipelines)

Usage:
    conda activate ascformer_env
    cd /Path/to/your/data/Manipulated_ID/RTM/ASCFormer
    python test_ascformer_id.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS

# Explicitly import mmseg.models to trigger registration of custom components
import mmseg.models 
from mmseg.models.data_preprocessor import SegDataPreProcessorWithExtra
from mmseg.models.segmentors.my_model_full import MyModelFull
print("Registered modules in MODELS segmentors:", [k for k in MODELS.module_dict.keys() if 'MyModel' in k or 'SegData' in k])

# ───────────────────── Paths ─────────────────────
ASCFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH    = os.path.join(ASCFORMER_ROOT, 'configs', 'ascformer', 'ascformer_rtm.py')
CKPT_PATH      = os.path.join(ASCFORMER_ROOT, 'ascformer_model.pth')

# Dataset roots (each has test/fake/ and test/real/)
STRATEGY_DIRS = {
    'Full_ID':  '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split',
    'Face':     '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_faces',
    'Content':  '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content',
    'Template': '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ───────────────── ELA & DCT Utilities ──────────────────
# These replicate the ELA and BlockDCT transforms from the ASCFormer pipeline

def compute_ela(img_bgr, quality=80):
    """Compute Error Level Analysis image."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recompressed = cv2.imread(tmp_path)
    finally:
        os.unlink(tmp_path)
    ela = cv2.absdiff(img_bgr, recompressed).astype(np.float32)
    return ela  # [H, W, 3] float32


def zigzag_indices():
    """Return zigzag scanning order for 8x8 block."""
    idx = np.zeros(64, dtype=int)
    row, col = 0, 0
    for i in range(64):
        idx[i] = row * 8 + col
        if (row + col) % 2 == 0:
            if col == 7:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:
            if row == 7:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1
    return idx


def compute_block_dct(img_bgr):
    """Compute block-wise DCT (single channel, zigzag reordered)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    # Pad to multiple of 8
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    padded = np.zeros((new_h, new_w), dtype=np.float32)
    padded[:h, :w] = gray

    # Block DCT
    dct_img = np.zeros_like(padded)
    for i in range(0, new_h, 8):
        for j in range(0, new_w, 8):
            block = padded[i:i+8, j:j+8]
            dct_img[i:i+8, j:j+8] = cv2.dct(block)

    # Zigzag reorder - pick first coefficient (DC) per block
    # Actually, based on the config, zigzag=True means we get single channel
    # and the DCTProcessor expects [1, H, W] input
    return dct_img[:h, :w]  # [H, W] float32


# ───────────────── Model Loading ──────────────────

def load_model():
    """Load ASCFormer model with pretrained weights via MMSeg config."""
    cfg = Config.fromfile(CONFIG_PATH)

    # Build model
    cfg.model.pretrained = None
    if hasattr(cfg.model.backbone, 'backbone_main'):
        cfg.model.backbone.backbone_main.pretrained = None
    if hasattr(cfg.model.backbone, 'backbone_extra'):
        cfg.model.backbone.backbone_extra.pretrained = None

    # Ensure custom modules are registered with correct scope
    if 'data_preprocessor' in cfg.model:
        if not cfg.model.data_preprocessor.type.startswith('mmseg.'):
            cfg.model.data_preprocessor.type = 'mmseg.' + cfg.model.data_preprocessor.type
    
    print(f"Building model of type: {cfg.model.get('type')}")
    model = MODELS.build(cfg.model)
    model.cfg = cfg

    # Load checkpoint
    checkpoint = load_checkpoint(model, CKPT_PATH, map_location='cpu')
    model.to(DEVICE).eval()
    print(f"✅ ASCFormer loaded on {DEVICE}")
    return model, cfg


# ───────────────── Inference ──────────────────

@torch.no_grad()
def predict_score(model, cfg, img_path):
    """
    Run ASCFormer on a single image and return a scalar detection score.

    Returns
    -------
    score : float in [0, 1]
        Higher ⇒ model thinks the image is tampered.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return 0.0

    h, w = img_bgr.shape[:2]

    # Compute forensic features
    ela = compute_ela(img_bgr, quality=80)        # [H, W, 3]
    dct = compute_block_dct(img_bgr)               # [H, W]

    # Prepare image tensor: normalize like SAM/SegFormer
    mean = np.array([123.675, 116.28, 103.53])
    std  = np.array([58.395, 57.12, 57.375])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_norm = (img_rgb - mean) / std

    # Pad to multiple of 32 (required by test_cfg size_divisor)
    div = 32
    new_h = ((h + div - 1) // div) * div
    new_w = ((w + div - 1) // div) * div

    # Image tensor [1, 3, H, W]
    img_pad = np.zeros((new_h, new_w, 3), dtype=np.float32)
    img_pad[:h, :w] = img_norm
    img_t = torch.from_numpy(img_pad.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    # ELA tensor [1, 3, H, W]
    ela_pad = np.zeros((new_h, new_w, 3), dtype=np.float32)
    ela_pad[:h, :w] = ela
    ela_t = torch.from_numpy(ela_pad.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    # DCT tensor [1, 1, H, W]
    dct_pad = np.zeros((new_h, new_w), dtype=np.float32)
    dct_pad[:h, :w] = dct
    dct_t = torch.from_numpy(dct_pad).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Build extras dict
    extras = {
        'dct': dct_t,
        'ela': ela_t,
    }

    # Build metadata
    batch_img_metas = [dict(
        ori_shape=(h, w),
        img_shape=(new_h, new_w),
        pad_shape=(new_h, new_w),
        padding_size=[0, new_w - w, 0, new_h - h],
        img_path=img_path,
    )]

    try:
        # Use whole_inference (test_cfg.mode = 'whole')
        seg_logits = model.whole_inference(img_t, extras, batch_img_metas)
        # seg_logits: [1, 2, H, W] — channel 0=background, channel 1=tampered
        prob_map = F.softmax(seg_logits, dim=1)[0, 1, :h, :w]  # [H, W]
        prob_np = prob_map.float().cpu().numpy()

        # Detection score: mean of top 5% pixel probabilities
        flat = prob_np.flatten()
        k = max(1, int(len(flat) * 0.05))
        top_k = np.partition(flat, -k)[-k:]
        score = float(top_k.mean())
    except Exception as e:
        print(f"  ⚠ Error processing {os.path.basename(img_path)}: {e}")
        score = 0.0

    return score


# ───────────────── Evaluation ──────────────────

def evaluate_strategy(model, cfg, strategy_name, data_root):
    """Evaluate one strategy (Full ID / Face / Content / Template)."""
    fake_dir = os.path.join(data_root, 'test', 'fake')
    real_dir = os.path.join(data_root, 'test', 'real')

    fake_imgs = sorted(glob(os.path.join(fake_dir, '*.jpg')) +
                        glob(os.path.join(fake_dir, '*.png')))
    real_imgs = sorted(glob(os.path.join(real_dir, '*.jpg')) +
                        glob(os.path.join(real_dir, '*.png')))

    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Fake images: {len(fake_imgs)}   Real images: {len(real_imgs)}")
    print(f"{'='*60}")

    if len(fake_imgs) == 0 and len(real_imgs) == 0:
        print("  ⚠ No images found — skipping.")
        return None

    all_scores = []
    all_labels = []    # Real=0, Fake=1

    # Process fake images
    print(f"  Processing FAKE images...")
    for p in tqdm(fake_imgs, desc=f"  [{strategy_name}] Fake"):
        score = predict_score(model, cfg, p)
        all_scores.append(score)
        all_labels.append(1)

    # Process real images
    print(f"  Processing REAL images...")
    for p in tqdm(real_imgs, desc=f"  [{strategy_name}] Real"):
        score = predict_score(model, cfg, p)
        all_scores.append(score)
        all_labels.append(0)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # ── Metrics ──
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
        fpr_curve, tpr_curve, thresholds = roc_curve(all_labels, all_scores)
        eer = brentq(lambda x : 1. - x - interp1d(fpr_curve, tpr_curve)(x), 0., 1.)
    else:
        auc = float('nan')
        eer = float('nan')

    preds = (all_scores >= 0.5).astype(int)
    overall_acc = accuracy_score(all_labels, preds) * 100

    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    real_acc = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
    fake_acc = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0.0

    print(f"\n  ┌────────────────────────────────────────┐")
    print(f"  │  Results: {strategy_name:<28s}  │")
    print(f"  ├────────────────────────────────────────┤")
    print(f"  │  Overall Accuracy : {overall_acc:6.2f}%             │")
    print(f"  │  AUC              : {auc:6.4f}              │")
    print(f"  │  EER              : {eer:6.4f}              │")
    print(f"  │  Real Accuracy    : {real_acc:6.2f}%             │")
    print(f"  │  Fake Accuracy    : {fake_acc:6.2f}%             │")
    print(f"  │  FPR              : {fpr:6.2f}%             │")
    print(f"  │  FNR              : {fnr:6.2f}%             │")
    print(f"  └────────────────────────────────────────┘")

    return {
        'strategy': strategy_name,
        'overall_acc': overall_acc,
        'auc': auc,
        'eer': eer,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'fpr': fpr,
        'fnr': fnr,
        'n_real': len(real_imgs),
        'n_fake': len(fake_imgs),
    }


# ───────────────── Main ──────────────────

def main():
    print("=" * 60)
    print("  ASCFormer Evaluation on ID Card Dataset")
    print("  Labels: Real=0, Fake=1")
    print("=" * 60)

    model, cfg = load_model()

    results = []
    for strategy_name, data_root in STRATEGY_DIRS.items():
        r = evaluate_strategy(model, cfg, strategy_name, data_root)
        if r is not None:
            results.append(r)

    # ── Summary Table ──
    print("\n\n" + "=" * 105)
    print("                                   ASCFormer — Summary Results")
    print("=" * 105)
    print(f"  {'Strategy':<12s} | {'Acc%':>8s} | {'AUC':>8s} | {'EER':>8s} | {'Real%':>8s} | {'Fake%':>8s} | {'FPR%':>8s} | {'FNR%':>8s} | {'#Real':>6s} | {'#Fake':>6s}")
    print("-" * 105)
    for r in results:
        print(f"  {r['strategy']:<12s} | {r['overall_acc']:7.2f}% | {r['auc']:8.4f} | {r['eer']:8.4f} | {r['real_acc']:7.2f}% | {r['fake_acc']:7.2f}% | {r['fpr']:7.2f}% | {r['fnr']:7.2f}% | {r['n_real']:>6d} | {r['n_fake']:>6d}")
    print("=" * 105)


if __name__ == '__main__':
    main()
