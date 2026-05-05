"""
Test IML-ViT (pretrained) on 4-Strategy ID Manipulation Dataset.
Evaluates all 3 checkpoints × 4 strategies = 12 evaluations.

IML-ViT is a localization model (outputs pixel-level mask).
We adapt it for binary classification by computing the mean
manipulation probability across the predicted mask as the score.
"""

import os
import sys
import io
import json
import time
import glob
import argparse
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout (debug prints from model forward)."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, confusion_matrix, roc_curve
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ─── IML-ViT imports ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iml_vit_model
from utils.iml_transforms import get_albu_transforms


# ─── Configuration ─────────────────────────────────────────────────
BASE = "/Path/to/your/data/Manipulated_ID"

STRATEGIES = {
    "Full_ID":      os.path.join(BASE, "Final_ID_dataset_split"),
    "Face_Only":    os.path.join(BASE, "Final_ID_dataset_split_faces"),
    "Template_Only":os.path.join(BASE, "Final_ID_dataset_split_template"),
    "Content_Only": os.path.join(BASE, "Final_ID_dataset_split_content"),
}

CHECKPOINTS = {
    "IML-ViT_default": "checkpoints/IML-ViT_checkpoints/iml-vit_checkpoint.pth",
    "IML-ViT_CASIAv2": "checkpoints/IML-ViT_checkpoints/iml-vit_checkpoint_casiav2_20231014.pth",
    "IML-ViT_TruFor":  "checkpoints/IML-ViT_checkpoints/iml-vit_checkpoint_trufor_20231104.pth",
}


# ─── Helpers ───────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    """Load IML-ViT model from a raw state-dict checkpoint."""
    with suppress_stdout():
        model = iml_vit_model.iml_vit_model(
            vit_pretrain_path=None,       # don't load MAE; we load full ckpt
            predict_head_norm="BN",
            edge_lambda=20,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # If checkpoint is wrapped in a 'model' key, unwrap it
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def collect_test_images(strategy_dir):
    """Collect (path, label) pairs from test/fake and test/real."""
    test_dir = os.path.join(strategy_dir, "test")
    items = []
    for label_name, label_val in [("fake", 1), ("real", 0)]:
        folder = os.path.join(test_dir, label_name)
        if not os.path.isdir(folder):
            print(f"  WARNING: {folder} not found, skipping.")
            continue
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                items.append((fpath, label_val))
    return items


def preprocess_image(img_path, pad_transform):
    """Load image and apply IML-ViT test preprocessing (pad to 1024, normalize)."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)  # H, W, C
    original_shape = img_np.shape[:2]  # (H, W)

    # Create a dummy mask (all zeros - authentic) for the padding transform
    dummy_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)

    # Apply padding + normalization + ToTensor
    res = pad_transform(image=img_np, masks=[dummy_mask])
    img_tensor = res["image"]          # C, H, W  (1024×1024)
    return img_tensor, original_shape


@torch.no_grad()
def inference_score(model, img_tensor, original_shape, device):
    """
    Run IML-ViT inference and return a scalar manipulation score.
    Score = mean of predicted manipulation probability over the
    original (non-padded) region.
    """
    img = img_tensor.unsqueeze(0).to(device)   # 1, 3, 1024, 1024

    # Create dummy masks for forward pass (model needs them for loss computation)
    dummy_mask = torch.zeros(1, 1, 1024, 1024, device=device)
    dummy_edge = torch.zeros(1, 1, 1024, 1024, device=device)
    shape_tensor = torch.tensor([original_shape], device=device)

    # Forward pass (suppress debug prints from model)
    with suppress_stdout():
        _, mask_pred, _ = model(img, dummy_mask, dummy_edge, shape_tensor)
    # mask_pred: (1, 1, 1024, 1024) after sigmoid

    # Extract only the original (non-padded) region
    h, w = original_shape
    region = mask_pred[0, 0, :h, :w]

    # Mean manipulation probability as detection score
    score = region.mean().item()
    return score


def find_optimal_threshold(y_true, y_scores):
    """Find threshold that maximizes accuracy."""
    thresholds = np.linspace(0, 1, 1001)
    best_acc = 0
    best_t = 0.5
    for t in thresholds:
        preds = (np.array(y_scores) >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc

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

def compute_metrics(y_true, y_scores, threshold=None):
    """Compute all metrics including per-class accuracy."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # AUC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0

    # Optimal threshold
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Per-class accuracy
    fake_mask = (y_true == 1)
    real_mask = (y_true == 0)
    fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask]) if fake_mask.sum() > 0 else 0.0
    real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask]) if real_mask.sum() > 0 else 0.0

    n_fake = int(fake_mask.sum())
    n_real = int(real_mask.sum())

    # Advanced metrics
    eer = compute_eer(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # labels: [real(0), fake(1)]
    tn, fp, fn, tp = cm.ravel()
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # real as fake
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # fake as real
    fpr = bpcer
    fnr = apcer

    return {
        "auc":       round(float(auc), 4),
        "accuracy":  round(float(acc), 4),
        "threshold": round(float(threshold), 4),
        "fake_acc":  round(float(fake_acc), 4),
        "real_acc":  round(float(real_acc), 4),
        "eer":       round(float(eer), 4),
        "bpcer":     round(float(bpcer), 4),
        "apcer":     round(float(apcer), 4),
        "fpr":       round(float(fpr), 4),
        "fnr":       round(float(fnr), 4),
        "n_fake":    n_fake,
        "n_real":    n_real,
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
    }


# ─── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test IML-ViT on ID dataset")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Test only this checkpoint name (e.g. 'IML-ViT_default')")
    parser.add_argument("--strategy", default=None, type=str,
                        help="Test only this strategy (e.g. 'Full_ID')")
    parser.add_argument("--max_images", default=None, type=int,
                        help="Max images per class for quick testing")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare the padding transform (test mode: no augmentation, just pad+normalize)
    pad_transform = get_albu_transforms("pad", outputsize=1024)

    # Filter checkpoints and strategies
    ckpts = CHECKPOINTS
    if args.checkpoint:
        ckpts = {k: v for k, v in CHECKPOINTS.items() if k == args.checkpoint}
    strats = STRATEGIES
    if args.strategy:
        strats = {k: v for k, v in STRATEGIES.items() if k == args.strategy}

    all_results = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for ckpt_name, ckpt_rel_path in ckpts.items():
        ckpt_path = os.path.join(script_dir, ckpt_rel_path)
        print(f"\n{'='*70}")
        print(f"Loading checkpoint: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"{'='*70}")

        model = load_model(ckpt_path, device)

        for strat_name, strat_dir in strats.items():
            print(f"\n  Strategy: {strat_name}")
            print(f"  {'─'*50}")

            # Collect test images
            items = collect_test_images(strat_dir)
            if not items:
                print(f"  No images found, skipping.")
                continue

            # Optionally limit
            if args.max_images:
                fake_items = [(p, l) for p, l in items if l == 1][:args.max_images]
                real_items = [(p, l) for p, l in items if l == 0][:args.max_images]
                items = fake_items + real_items

            n_fake = sum(1 for _, l in items if l == 1)
            n_real = sum(1 for _, l in items if l == 0)
            print(f"  Images: {len(items)} total (fake={n_fake}, real={n_real})")

            # Run inference
            y_true = []
            y_scores = []
            t0 = time.time()

            for idx, (img_path, label) in enumerate(items):
                try:
                    img_tensor, orig_shape = preprocess_image(img_path, pad_transform)
                    score = inference_score(model, img_tensor, orig_shape, device)
                    y_true.append(label)
                    y_scores.append(score)
                except Exception as e:
                    print(f"    ERROR on {os.path.basename(img_path)}: {e}")
                    continue

                if (idx + 1) % 500 == 0 or (idx + 1) == len(items):
                    elapsed = time.time() - t0
                    rate = (idx + 1) / elapsed
                    print(f"    [{idx+1}/{len(items)}] {rate:.1f} img/s, elapsed={elapsed:.1f}s")

            elapsed = time.time() - t0
            print(f"  Inference done in {elapsed:.1f}s ({len(y_true)}/{len(items)} images)")

            # Compute metrics
            metrics = compute_metrics(y_true, y_scores)
            metrics["checkpoint"] = ckpt_name
            metrics["strategy"]   = strat_name
            metrics["time_sec"]   = round(elapsed, 1)
            all_results.append(metrics)

            # Print summary
            print(f"  Results:")
            print(f"    AUC      = {metrics['auc']:.4f}")
            print(f"    Accuracy = {metrics['accuracy']:.4f}  (threshold={metrics['threshold']:.3f})")
            print(f"    Fake Acc = {metrics['fake_acc']:.4f}  ({metrics['TP']}/{metrics['n_fake']})")
            print(f"    Real Acc = {metrics['real_acc']:.4f}  ({metrics['TN']}/{metrics['n_real']})")
            print(f"    EER      = {metrics['eer']:.4f}")
            print(f"    FPR      = {metrics['fpr']:.4f} | FNR = {metrics['fnr']:.4f}")

        # Free GPU memory before next checkpoint
        del model
        torch.cuda.empty_cache()

    # ─── Final Summary Table ───────────────────────────────────────
    print("="*120)
    print("FINAL SUMMARY — IML-ViT on ID Manipulation Detection")
    print("="*120)
    header = f"{'Checkpoint':<20} {'Strategy':<16} {'AUC':>6} {'Acc':>6} {'EER':>6} {'FPR':>6} {'FNR':>6} {'FakeAcc':>8} {'RealAcc':>8}"
    print(header)
    print("-"*120)
    for r in all_results:
        row = (f"{r['checkpoint']:<20} {r['strategy']:<16} "
               f"{r['auc']:>6.4f} {r['accuracy']:>6.4f} {r['eer']:>6.4f} "
               f"{r['fpr']:>6.4f} {r['fnr']:>6.4f} "
               f"{r['fake_acc']:>8.4f} {r['real_acc']:>8.4f}")
        print(row)
    print("="*120)

    # ─── Save JSON ─────────────────────────────────────────────────
    out_json = os.path.join(script_dir, "iml_vit_id_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {out_json}")


if __name__ == "__main__":
    main()
