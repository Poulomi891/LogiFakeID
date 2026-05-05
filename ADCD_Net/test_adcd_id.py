"""
Evaluate pre-trained ADCD-Net on the ID card dataset across four
regional strategies: Full ID, Face, Template, Content.

ADCD-Net is a *localization* model (pixel-level forgery mask prediction).
For *detection* (image-level real/fake classification), we derive a score
from the predicted localization map:
    score = max predicted manipulation probability across pixels
    → high score ⇒ model thinks the image is tampered

Metrics reported:
    - Overall Accuracy
    - AUC (Area Under ROC Curve)
    - Per-Class Accuracy  (Real accuracy, Fake accuracy)

Labels:  Real = 0,  Fake = 1  (consistent with your other pipelines)
"""

import os
import sys
import pickle
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from copy import deepcopy
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ───────────────────── Paths ─────────────────────
ADCD_ROOT   = os.path.dirname(os.path.abspath(__file__))
EXP_DATA    = os.path.join(ADCD_ROOT, 'ADCD-Net_exp_data')
CKPT_PATH   = os.path.join(EXP_DATA, 'ADCDNet.pth')
DOCRES_PATH = os.path.join(EXP_DATA, 'docres.pkl')
QT_PATH     = os.path.join(EXP_DATA, 'qt_table.pk')

# Dataset roots  (each has test/fake/ and test/real/)
STRATEGY_DIRS = {
    'Full_ID':  '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split',
    'Face':     '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_faces',
    'Content':  '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content',
    'Template': '/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_MAX_SIZE = 512          # longest edge resize
IMG_MEAN = (0.485, 0.455, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

# ───────────────── Utility Functions ──────────────────

class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle NumPy core module changes
        if module == "numpy.core.numeric":
            module = "numpy._core.numeric"
        elif module == "numpy.core.multiarray":
            module = "numpy._core.multiarray"
        elif module == "numpy.core.umath":
            module = "numpy._core.umath"
            
        # Try to dynamically import the module if it's not already in sys.modules
        if module not in sys.modules:
            try:
                __import__(module)
            except ImportError:
                # If we fail to import numpy._core.X, try numpy.core.X (for NumPy 1.x)
                fallback = module.replace("numpy._core", "numpy.core")
                try:
                    __import__(fallback)
                    module = fallback
                except ImportError:
                    pass
        
        return super().find_class(module, name)

def load_qt(qt_path):
    """Load {quality_factor: 8×8 tensor} dictionary."""
    # Ensure submodules are loaded to prevent pickle from failing
    try: import numpy._core.numeric; import numpy._core.multiarray; import numpy._core.umath
    except ImportError: pass
    try: import numpy.core.numeric; import numpy.core.multiarray; import numpy.core.umath
    except ImportError: pass

    with open(qt_path, 'rb') as f:
        try:
            pks_ = CompatibilityUnpickler(f).load()
        except Exception:
            f.seek(0)
            pks_ = pickle.load(f)
            
    return {k: torch.LongTensor(v) for k, v in pks_.items()}


def img_to_tensor(img_pil):
    """PIL Image → normalised float32 tensor [C,H,W]."""
    arr = np.array(img_pil).astype(np.float32) / 255.0
    arr = (arr - np.array(IMG_MEAN)) / np.array(IMG_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def extract_dct(img_pil):
    """Re-compress image at quality 100 and extract the DCT Y-channel."""
    from jpeg2dct.numpy import load as dct_load
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        grey = img_pil.convert("L")
        grey.save(tmp_path, "JPEG", quality=100)
        reopened = Image.open(tmp_path).convert('RGB')
        dct_y, _, _ = dct_load(tmp_path, normalized=False)
    finally:
        os.unlink(tmp_path)

    rows, cols, _ = dct_y.shape
    dct = np.empty((8 * rows, 8 * cols), dtype=np.int32)
    for j in range(rows):
        for i in range(cols):
            dct[8*j:8*(j+1), 8*i:8*(i+1)] = dct_y[j, i].reshape(8, 8)
    return dct, reopened


def pad_to_multiple(t, divisor=16):
    """Pad a [C,H,W] or [H,W] tensor so H and W are divisible by `divisor`."""
    if t.dim() == 3:
        _, h, w = t.shape
    else:
        h, w = t.shape
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    new_s = max(new_h, new_w)    # make square
    pad_h, pad_w = new_s - h, new_s - w
    if t.dim() == 3:
        return F.pad(t, (0, pad_w, 0, pad_h), value=0.0), h, w
    else:
        return F.pad(t, (0, pad_w, 0, pad_h), value=0), h, w


def resize_if_needed(img_cv2, max_size):
    """Resize so longest edge ≤ max_size, preserving aspect ratio."""
    h, w = img_cv2.shape[:2]
    if max(h, w) <= max_size:
        return img_cv2
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_eer(labels, scores):
    """Equal Error Rate calculation."""
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# ───────────────── Model Loading ──────────────────

def load_model():
    """Load ADCD-Net with pretrained weights (single-GPU, no DDP)."""
    # Temporarily patch cfg so the model __init__ can find docres
    import cfg
    cfg.docres_ckpt_path = DOCRES_PATH

    from model.model import ADCDNet
    model = ADCDNet()

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    state = ckpt['model']
    # Strip 'module.' prefix from DDP checkpoint
    clean = {k.replace('module.', ''): v for k, v in state.items()}
    miss, unexp = model.load_state_dict(clean, strict=False)
    if miss:
        print(f"⚠ Missing keys: {miss[:5]}{'...' if len(miss)>5 else ''}")
    if unexp:
        print(f"⚠ Unexpected keys: {unexp[:5]}{'...' if len(unexp)>5 else ''}")

    model.to(DEVICE).eval()
    print(f"✅ Model loaded on {DEVICE}")
    return model


# ───────────────── Inference ──────────────────

@torch.no_grad()
def predict_score(model, img_path, qts):
    """
    Run ADCD-Net on a single image and return a scalar detection score.

    Returns
    -------
    score : float in [0, 1]
        Higher ⇒ model thinks the image is tampered.
    """
    # Read & resize
    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None:
        return 0.0
    img_cv2 = resize_if_needed(img_cv2, VAL_MAX_SIZE)
    h, w = img_cv2.shape[:2]

    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

    # DCT extraction (re-compress at q=100)
    dct, img_pil_reopened = extract_dct(img_pil)
    qt = qts[100]   # quality factor 100

    # Image tensor
    img_t = img_to_tensor(img_pil_reopened)                         # [3,H,W]

    # OCR mask: all zeros (no text segmentation — valid for face crops;
    # for text-heavy regions the pp_map will just treat entire image as background)
    ocr_mask = torch.zeros(1, h, w, dtype=torch.long)

    # Ground-truth mask placeholder (not used during inference)
    mask = torch.zeros(1, h, w, dtype=torch.long)

    # Pad everything to multiple of 16, square
    img_t, orig_h, orig_w = pad_to_multiple(img_t, 16)
    dct_t = torch.tensor(np.clip(np.abs(dct), 0, 20), dtype=torch.long)
    dct_t, _, _ = pad_to_multiple(dct_t, 16)
    mask, _, _ = pad_to_multiple(mask.squeeze(0), 16)
    mask = mask.unsqueeze(0)
    ocr_mask, _, _ = pad_to_multiple(ocr_mask.squeeze(0), 16)
    ocr_mask = ocr_mask.unsqueeze(0)

    # Batch dim
    img_t     = img_t.unsqueeze(0).to(DEVICE)          # [1,3,S,S]
    dct_t     = dct_t.unsqueeze(0).long().to(DEVICE)    # [1,S,S]
    qt_t      = qt.unsqueeze(0).to(DEVICE)               # [1,8,8]
    mask      = mask.unsqueeze(0).to(DEVICE)              # [1,1,S,S]
    ocr_mask  = ocr_mask.unsqueeze(0).to(DEVICE)          # [1,1,S,S]

    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(img_t, dct_t, qt_t, mask, ocr_mask, is_train=False)[0]

    # logits: [1, 2, S, S] — channel 0=clean, channel 1=tampered
    prob_map = F.softmax(logits, dim=1)[0, 1, :orig_h, :orig_w]  # [H, W]
    prob_np = prob_map.float().cpu().numpy()

    # Detection score: max tamper probability across all pixels
    score = float(prob_np.max())
    return score


# ───────────────── Evaluation ──────────────────

def evaluate_strategy(model, strategy_name, data_root, qts):
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
        score = predict_score(model, p, qts)
        all_scores.append(score)
        all_labels.append(1)

    # Process real images
    print(f"  Processing REAL images...")
    for p in tqdm(real_imgs, desc=f"  [{strategy_name}] Real"):
        score = predict_score(model, p, qts)
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

    # EER
    eer = compute_eer(all_labels, all_scores)

    # Per-class accuracy and errors
    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    # cm: [TN, FP], [FN, TP]
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
    print("  ADCD-Net Evaluation on ID Card Dataset")
    print("  Labels: Real=0, Fake=1")
    print("=" * 60)

    # Load QT tables
    print("📦 Loading quantization tables...")
    qts = load_qt(QT_PATH)
    print(f"   Loaded {len(qts)} QT entries")

    # Load model
    print("📦 Loading ADCD-Net model...")
    model = load_model()

    # Evaluate each strategy
    results = []
    for strategy_name, data_root in STRATEGY_DIRS.items():
        r = evaluate_strategy(model, strategy_name, data_root, qts)
        if r is not None:
            results.append(r)

    # ── Summary Table ──
    print("\n\n" + "=" * 90)
    print("                    ADCD-Net — Summary Results")
    print("=" * 90)
    print(f"  {'Strategy':<12s} | {'Acc%':>8s} | {'AUC':>8s} | {'EER':>8s} | {'FPR%':>7s} | {'FNR%':>7s} | {'#Real':>6s} | {'#Fake':>6s}")
    print("-" * 90)
    for r in results:
        print(f"  {r['strategy']:<12s} | {r['overall_acc']:7.2f}% | {r['auc']:8.4f} | {r['eer']:8.4f} | {r['fpr']:6.2f}% | {r['fnr']:6.2f}% | {r['n_real']:>6d} | {r['n_fake']:>6d}")
    print("=" * 90)


if __name__ == '__main__':
    main()
