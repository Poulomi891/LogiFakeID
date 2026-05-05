import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import warnings

# ── Dynamic fix for libstdc++ version mismatch (Affects Dlib/TruFor) ──────────
CONDA_LIB = "/home/tbvl_lab/miniconda3/envs/trufor_env/lib"
if CONDA_LIB not in os.environ.get("LD_LIBRARY_PATH", "") and "RE_EXECED" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = CONDA_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["RE_EXECED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = "/Path/to/your/data/Manipulated_ID"
TRUFOR_ROOT = os.path.join(REPO_ROOT, "TruFor", "TruFor_train_test")
sys.path.insert(0, TRUFOR_ROOT)

try:
    from lib.config import config, update_config
    from lib.utils import get_model
except ImportError:
    print("❌ Error: Could not import TruFor modules. Check TRUFOR_ROOT and PYTHONPATH.")
    sys.exit(1)

# ── Strategies ───────────────────────────────────────────────────────────────
DATASETS = {
    "Full_ID":          os.path.join(REPO_ROOT, "Final_ID_dataset_split", "test"),
    "Face_Crops":       os.path.join(REPO_ROOT, "Final_ID_dataset_split_faces", "test"),
    "Template_Visible": os.path.join(REPO_ROOT, "Final_ID_dataset_split_content", "test"),
    "Content_Visible":  os.path.join(REPO_ROOT, "Final_ID_dataset_split_template", "test")
}

MODEL_PATH = os.path.join(TRUFOR_ROOT, "pretrained_models", "trufor.pth.tar")
RESULTS_FILE = "results_trufor_pretrained.txt"

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

def evaluate_strategy(strategy_name, test_root):
    print(f"\n🚀 Evaluating Strategy: {strategy_name}")
    print(f"   📂 Path: {test_root}")
    
    # ── Model Loading ────────────────────────────────────────────────────────
    # TruFor uses a config based loading
    # We update config to point to the pretrained weights
    config.defrost()
    config.merge_from_file(os.path.join(TRUFOR_ROOT, 'lib/config/trufor_ph3.yaml'))
    config.TEST.MODEL_FILE = MODEL_PATH
    config.freeze()
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = get_model(config)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    scores = []
    labels = []
    
    # Subfolders: real, fake
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = os.path.join(test_root, label_name)
        if not os.path.exists(folder):
            print(f"   ⚠️ Warning: Folder not found: {folder}")
            continue
            
        img_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"   🔍 Found {len(img_files)} images in {label_name}")
        
        for fname in tqdm(img_files, desc=f"      {label_name}"):
            img_path = os.path.join(folder, fname)
            try:
                # TruFor takes raw RGB tensors [1, 3, H, W]
                # Note: TruFor's TestDataset does not use specific normalization in its __getitem__
                # but models within might. Let's look at TruFor's TestDataset implementation.
                img = Image.open(img_path).convert("RGB")
                img_t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    # TruFor forward: pred, conf, det, npp
                    _, _, det, _ = model(img_t)
                    # Global integrity score
                    score = torch.sigmoid(det).item()
                
                scores.append(score)
                labels.append(label_val)
            except Exception as e:
                print(f"      ❌ Error processing {fname}: {e}")

    if not scores:
        return None

    # ── Metrics ──────────────────────────────────────────────────────────────
    auc = roc_auc_score(labels, scores)
    
    # Simple binary accuracy at 0.5 threshold
    preds = [1 if s > 0.5 else 0 for s in scores]
    acc = accuracy_score(labels, preds)
    
    report = classification_report(labels, preds, target_names=["Real", "Fake"], output_dict=True)
    real_acc = report["Real"]["precision"] # Precision for Real class acts as Acc if labels are balanced
    fake_acc = report["Fake"]["recall"]    # This is standard per-class acc
    
    # Better way for per-class Accuracy:
    labels_np = np.array(labels)
    preds_np = np.array(preds)
    acc_real = np.mean(preds_np[labels_np == 0] == 0)
    acc_fake = np.mean(preds_np[labels_np == 1] == 1)
    
    # Calculate advanced metrics
    eer = compute_eer(labels, scores)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    # labels: [real(0), fake(1)]
    tn, fp, fn, tp = cm.ravel()
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # real as fake
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # fake as real
    fpr = bpcer
    fnr = apcer
    
    print(f"   ✅ AUC: {auc:.4f} | Acc: {acc:.4f} | EER: {eer:.4f}")
    print(f"   ✅ Per-Class: Real {acc_real:.4f}, Fake {acc_fake:.4f}")
    print(f"   ✅ FPR (BPCER): {fpr:.4f}, FNR (APCER): {fnr:.4f}")
    
    return {
        "auc": auc,
        "acc": acc,
        "acc_real": acc_real,
        "acc_fake": acc_fake,
        "eer": eer,
        "fpr": fpr,
        "fnr": fnr
    }

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model weights not found at {MODEL_PATH}")
        sys.exit(1)
        
    all_results = {}
    
    with open(RESULTS_FILE, "w") as f:
        f.write("TruFor Pretrained Evaluation Results\n")
        f.write("====================================\n\n")
        
        for name, path in DATASETS.items():
            res = evaluate_strategy(name, path)
            if res:
                all_results[name] = res
                f.write(f"Strategy: {name}\n")
                f.write(f"  AUC: {res['auc']:.4f}\n")
                f.write(f"  Accuracy: {res['acc']:.4f}\n")
                f.write(f"  Real Accuracy: {res['acc_real']:.4f}\n")
                f.write(f"  Fake Accuracy: {res['acc_fake']:.4f}\n")
                f.write(f"  EER: {res['eer']:.4f}\n")
                f.write(f"  FPR (BPCER): {res['fpr']:.4f}\n")
                f.write(f"  FNR (APCER): {res['fnr']:.4f}\n")
                f.write("-" * 30 + "\n")
                f.flush()

    print(f"\n✨ All evaluations complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
