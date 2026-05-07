import os
import sys

# ── Dynamic fix for libstdc++ version mismatch ───────────────────────────────
CONDA_LIB = "/home/tbvl_lab/miniconda3/envs/reface_env/lib"
if CONDA_LIB not in os.environ.get("LD_LIBRARY_PATH", "") and "RE_EXECED" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = CONDA_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["RE_EXECED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
try:
    import dlib
    HAS_DLIB = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import dlib ({e}). Visualization will be disabled.")
    HAS_DLIB = False
import warnings

warnings.filterwarnings("ignore")

# ── Paths & Setup ────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/Effort-AIGI-Detection"
BENCH_ROOT = os.path.join(REPO_ROOT, "DeepfakeBench")
sys.path.insert(0, BENCH_ROOT)
sys.path.insert(0, os.path.join(BENCH_ROOT, "training"))

from training.detectors import DETECTOR

CONTENT_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content"
TEMPLATE_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template"
PRETRAINED_WEIGHTS = os.path.join(REPO_ROOT, "effort_clip_L14_trainOn_FaceForensic.pth")
CONFIG_PATH = os.path.join(BENCH_ROOT, "training/config/detector/effort.yaml")
RESULTS_DIR = "/Path/to/your/data/Manipulated_ID/Results/results_effort_pretrained_regional"
VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

if HAS_DLIB:
    FACE_DETECTOR = dlib.get_frontal_face_detector()
else:
    FACE_DETECTOR = None

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

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_face_crop(pil_img, margin=0.2):
    if not HAS_DLIB or FACE_DETECTOR is None: return None, None
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    rects = FACE_DETECTOR(gray, 0)
    if not rects: return None, None
    rect = max(rects, key=lambda r: r.area())
    w, h = rect.width(), rect.height()
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    x1 = max(0, int(x1 - margin * w)); y1 = max(0, int(y1 - margin * h))
    x2 = min(img_cv.shape[1], int(x2 + margin * w)); y2 = min(img_cv.shape[0], int(y2 + margin * h))
    return pil_img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)

class EffortRegionalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            folder = os.path.join(root, label_name)
            if not os.path.isdir(folder): continue
            for fname in os.listdir(folder):
                self.samples.append((os.path.join(folder, fname), label_idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, label, path
        except:
            return torch.zeros(3, 224, 224), label, path

# ── Visualization ─────────────────────────────────────────────────────────────
def visualize_focus(model, img_path, gt, pr, suffix, strategy, device):
    img_full = Image.open(img_path).convert("RGB")
    
    # Try to find the face region for visualization if it's not masked
    crop, bbox = get_face_crop(img_full)
    if crop is None:
        input_img = img_full
        bbox = (0, 0, img_full.size[0], img_full.size[1])
    else:
        input_img = crop

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    input_t = transform(input_img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.backbone(input_t, output_attentions=True)
        # Attention roll-out from last layer
        attn = outputs.attentions[-1][0, :, 0, 1:].mean(dim=0)
        attn = attn.view(16, 16).cpu().numpy()
    
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    res_heatmap = cv2.resize(attn, (bbox[2] - bbox[0], bbox[3] - bbox[1]), interpolation=cv2.INTER_LINEAR)
    full_heatmap = np.zeros((img_full.size[1], img_full.size[0]), dtype=np.float32)
    full_heatmap[bbox[1]:bbox[3], bbox[0]:bbox[2]] = res_heatmap
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img_full)
    plt.imshow(full_heatmap, cmap='jet', alpha=0.5)
    plt.axis("off")
    gt_txt = "Real" if gt == 0 else "Fake"
    pr_txt = "Real" if pr == 0 else "Fake"
    plt.title(f"Effort {strategy} Focus: (GT:{gt_txt}, Pred:{pr_txt})")
    
    save_path = os.path.join(VIS_DIR, f"focus_pretrained_{strategy}_{suffix}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# ── Test Strategy ─────────────────────────────────────────────────────────────
def test_regional_strategy(strategy):
    print(f"\n🚀 Testing PRETRAINED Effort | Region: {strategy}")
    data_dir = CONTENT_ROOT if strategy == "content_masked" else TEMPLATE_ROOT
    
    # 1. Load Model
    with open(CONFIG_PATH, "r") as f: config = yaml.safe_load(f)
    model = DETECTOR[config.get("detector_type", "effort")](config).to(DEVICE)
    
    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. Dataset & Loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    dataset = EffortRegionalDataset(os.path.join(data_dir, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 3. Evaluation
    all_preds, all_labels, all_probs, all_paths = [], [], [], []
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Eval"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            output = model({"image": imgs}, inference=True)
            logits = output["cls"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    eer = compute_eer(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    # labels: [real(0), fake(1)]
    tn, fp, fn, tp = cm.ravel()
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # real as fake
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # fake as real
    fpr, fnr = bpcer, apcer
    
    print(f"📊 Results for {strategy}:")
    print(f"   Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")
    print(f"   FPR: {fpr:.4f} | FNR: {fnr:.4f}")
    
    cls_accs = []
    class_names = ["real", "fake"]
    for idx, cls_name in enumerate(class_names):
        mask = (all_labels == idx)
        c_acc = (all_preds[mask] == idx).sum() / mask.sum() if mask.sum() > 0 else 0.0
        cls_accs.append(c_acc)
        print(f"    {cls_name:<12} : {c_acc*100:.2f}%")

    # Save to results file
    results_file = os.path.join(RESULTS_DIR, "effort_pretrained_regional_results.txt")
    with open(results_file, "a") as f:
        f.write(f"Effort_Pretrained,{strategy},{acc:.4f},{auc:.4f},{eer:.4f},{bpcer:.4f},{apcer:.4f},{fpr:.4f},{fnr:.4f},{cls_accs[0]:.4f},{cls_accs[1]:.4f}\n")

    # Misclass Report
    misclass_file = os.path.join(RESULTS_DIR, f"misclass_{strategy}.txt")
    with open(misclass_file, "w") as f:
        f.write("Path,GT,Pred\n")
        m_idx, c_idx = None, None
        for i in range(len(all_labels)):
            if all_preds[i] != all_labels[i]:
                f.write(f"{all_paths[i]},{all_labels[i]},{all_preds[i]}\n")
                if m_idx is None: m_idx = i
            elif c_idx is None:
                c_idx = i
    
    # Visualization
    if c_idx is not None: visualize_focus(model, all_paths[c_idx], int(all_labels[c_idx]), int(all_preds[c_idx]), "correct", strategy, DEVICE)
    if m_idx is not None: visualize_focus(model, all_paths[m_idx], int(all_labels[m_idx]), int(all_preds[m_idx]), "wrong", strategy, DEVICE)

if __name__ == "__main__":
    res_file = os.path.join(RESULTS_DIR, "effort_pretrained_regional_results.txt")
    with open(res_file, "w") as f:
        f.write("Model,Strategy,Accuracy,AUC,EER,BPCER,APCER,FPR,FNR,Acc_real,Acc_fake\n")

    for strat in ["content_masked", "template_masked"]:
        test_regional_strategy(strat)
    print("\n✅ Regional Pretrained Effort testing complete.")
