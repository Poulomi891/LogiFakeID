import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

"""
Evaluate pre-trained SAFIRE on the ID card dataset across four
regional strategies: Full ID, Face, Template, Content.

SAFIRE (Segment Any Forged Image Region) is a *localization* model built
on top of Segment Anything Model (SAM). It outputs a pixel-level forgery
probability heatmap via binary localization.

For *detection* (image-level real/fake classification), we derive a score
from the predicted heatmap:
    score = mean of top-k% pixel probabilities
    → high score ⇒ model thinks the image is tampered

Metrics reported:
    - Overall Accuracy
    - AUC (Area Under ROC Curve)
    - Per-Class Accuracy  (Real accuracy, Fake accuracy)

Labels:  Real = 0,  Fake = 1  (consistent with your other pipelines)

Usage:
    python test_safire_id.py

Requires: id_env environment with monai, torch, etc.
"""

import os
import sys
import numpy as np
import torch
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ───────────────────── Paths ─────────────────────
SAFIRE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAFIRE_ROOT)

WEIGHTS_DIR = os.path.join(SAFIRE_ROOT, 'SAFIRE-20260421T172202Z-3-001', 'SAFIRE')
SAFIRE_CKPT = os.path.join(WEIGHTS_DIR, 'safire.pth')
SAM_CKPT    = os.path.join(WEIGHTS_DIR, 'sam_vit_b_01ec64.pth')

# Dataset roots (each has test/fake/ and test/real/)
STRATEGY_DIRS = {
    'Full_ID':  '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split',
    'Face':     '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_faces',
    'Content':  '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content',
    'Template': '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ───────────────── Model Loading ──────────────────

def load_model():
    """Load SAFIRE model with pretrained weights."""
    from segment_anything import sam_model_registry
    from networks.safire_model import AdaptorSAM
    from networks.safire_predictor_binary import SafirePredictor

    print(f"📦 Loading SAM backbone from {SAM_CKPT}...")
    sam_model = sam_model_registry["vit_b_adaptor"](checkpoint=SAM_CKPT)

    safire_model = AdaptorSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).cuda()

    print(f"📦 Loading SAFIRE weights from {SAFIRE_CKPT}...")
    checkpoint = torch.load(SAFIRE_CKPT, map_location='cpu', weights_only=False)
    safire_model.load_state_dict(
        {k.replace("module.", ""): checkpoint["model"][k] for k in checkpoint["model"]}
    )
    print(f"   Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")

    # Create the automatic predictor (16×16 grid of point prompts)
    safire_predictor = SafirePredictor(
        safire_model,
        points_per_side=16,
        points_per_batch=64 * 4,
        pred_iou_thresh=0,
        stability_score_thresh=0.0,
        box_nms_thresh=0.0,
    )

    safire_model.eval()
    print(f"✅ SAFIRE model loaded on {DEVICE}")
    return safire_model, safire_predictor

# ── Metrics Helper ───────────────────────────────────────────────────────────
def compute_eer(labels, scores):
    """Compute Equal Error Rate (EER) given labels and scores."""
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except:
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer


# ───────────────── Inference ──────────────────

@torch.no_grad()
def predict_score(safire_predictor, img_path):
    """
    Run SAFIRE binary localization on a single image and return a
    scalar detection score.

    Returns
    -------
    score : float in [0, 1]
        Higher ⇒ model thinks the image is tampered.
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 1024×1024 (SAM's expected input size)
    img_resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    try:
        anns, safire_pred, max_confidence_indices = safire_predictor.safire_predict(img_resized)
    except Exception as e:
        print(f"  ⚠ Error processing {os.path.basename(img_path)}: {e}")
        return 0.0

    # safire_pred is a [1024, 1024] float array, values in [0, 1]
    # Higher values = higher probability of forgery

    # Detection score: mean of top 5% pixel probabilities
    # (more robust than raw max which can be noisy)
    flat = safire_pred.flatten()
    k = max(1, int(len(flat) * 0.05))
    top_k = np.partition(flat, -k)[-k:]
    score = float(top_k.mean())

    return score


# ───────────────── Evaluation ──────────────────

def evaluate_strategy(safire_model, safire_predictor, strategy_name, data_root):
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
        score = predict_score(safire_predictor, p)
        all_scores.append(score)
        all_labels.append(1)

    # Process real images
    print(f"  Processing REAL images...")
    for p in tqdm(real_imgs, desc=f"  [{strategy_name}] Real"):
        score = predict_score(safire_predictor, p)
        all_scores.append(score)
        all_labels.append(0)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # ── Metrics ──
    # AUC
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
    else:
        auc = float('nan')

    # Threshold at 0.5 for binary predictions
    preds = (all_scores >= 0.5).astype(int)
    overall_acc = accuracy_score(all_labels, preds) * 100

    # Per-class accuracy
    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    real_acc = (cm[0, 0] / cm[0].sum() * 100) if cm[0].sum() > 0 else 0.0
    fake_acc = (cm[1, 1] / cm[1].sum() * 100) if cm[1].sum() > 0 else 0.0

    # Calculate advanced metrics
    eer = compute_eer(all_labels, all_scores)
    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    # labels: [real(0), fake(1)]
    tn, fp, fn, tp = cm.ravel()
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # real as fake
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # fake as real
    fpr = bpcer
    fnr = apcer

    print(f"\n  ┌────────────────────────────────────────┐")
    print(f"  │  Results: {strategy_name:<28s}  │")
    print(f"  ├────────────────────────────────────────┤")
    print(f"  │  Overall Accuracy : {overall_acc:6.2f}%             │")
    print(f"  │  AUC              : {auc:6.4f}              │")
    print(f"  │  EER              : {eer:6.4f}              │")
    print(f"  │  Real Accuracy    : {real_acc:6.2f}%             │")
    print(f"  │  Fake Accuracy    : {fake_acc:6.2f}%             │")
    print(f"  │  FPR (BPCER)      : {fpr:6.4f}              │")
    print(f"  │  FNR (APCER)      : {fnr:6.4f}              │")
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
    print("  SAFIRE Evaluation on ID Card Dataset")
    print("  Labels: Real=0, Fake=1")
    print("=" * 60)

    # Load model
    safire_model, safire_predictor = load_model()

    # Evaluate each strategy
    results = []
    for strategy_name, data_root in STRATEGY_DIRS.items():
        r = evaluate_strategy(safire_model, safire_predictor, strategy_name, data_root)
        if r is not None:
            results.append(r)

    # ── Summary Table ──
    print("\n\n" + "=" * 100)
    print("                     SAFIRE — Summary Results")
    print("=" * 100)
    print(f"  {'Strategy':<12s} | {'Acc%':>8s} | {'AUC':>8s} | {'EER':>8s} | {'Real%':>8s} | {'Fake%':>8s} | {'FPR':>8s} | {'FNR':>8s}")
    print("-" * 100)
    for r in results:
        print(f"  {r['strategy']:<12s} | {r['overall_acc']:7.2f}% | {r['auc']:8.4f} | {r['eer']:8.4f} | {r['real_acc']:7.2f}% | {r['fake_acc']:7.2f}% | {r['fpr']:8.4f} | {r['fnr']:8.4f}")
    print("=" * 100)


if __name__ == '__main__':
    main()
