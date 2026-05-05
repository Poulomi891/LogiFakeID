import os
import sys

# ── Dynamic fix for libstdc++ version mismatch (affects dlib) ─────────────────
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
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import dlib
import warnings

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/Effort-AIGI-Detection"
BENCH_ROOT = os.path.join(REPO_ROOT, "DeepfakeBench")
sys.path.insert(0, BENCH_ROOT)
sys.path.insert(0, os.path.join(BENCH_ROOT, "training"))

from training.detectors import DETECTOR

SRC_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split"
FACE_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_faces"
CONFIG_PATH = os.path.join(BENCH_ROOT, "training/config/detector/effort.yaml")
WEIGHTS_DIR = "/Path/to/your/data/Manipulated_ID/weights/effort"
VIS_DIR = "/Path/to/your/data/Manipulated_ID/Results/results_effort_finetuned/visualizations_effort_finetuned"
os.makedirs(VIS_DIR, exist_ok=True)

FACE_DETECTOR = dlib.get_frontal_face_detector()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_face_crop(pil_img, margin=0.2):
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

class EffortEvalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            folder = os.path.join(root, label_name)
            if not os.path.isdir(folder): continue
            for fname in os.listdir(folder):
                if label_name == "fake" and fname.startswith("real_"): continue
                self.samples.append((os.path.join(folder, fname), label_idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label, path

# ── Visualization ─────────────────────────────────────────────────────────────
def visualize_focus(model, img_path, gt, pr, suffix, strategy, device):
    img_full = Image.open(img_path).convert("RGB")
    if strategy == "face":
        # For face strategy, the model was trained on crops.
        # We process the crop directly.
        input_img = img_full
        bbox = (0, 0, img_full.size[0], img_full.size[1])
    else:
        # For full strategy, we typically focus on the face area for saliency
        # But since it was trained on full ID, we could show full saliency.
        # However, following the repo's style, we'll map face attention back.
        crop, bbox = get_face_crop(img_full)
        if crop is None:
            input_img = img_full
            bbox = (0, 0, img_full.size[0], img_full.size[1])
        else:
            input_img = crop

    # Prepare for model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    input_t = transform(input_img).unsqueeze(0).to(device)
    
    model.eval()
    outputs = model.backbone(input_t, output_attentions=True)
    attn = outputs.attentions[-1][0, :, 0, 1:].mean(dim=0) # CLS attention to all 256 patches
    attn = attn.view(16, 16).cpu().detach().numpy()
    
    # Normalize Heatmap
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    
    # Map back to full image (or crop)
    res_heatmap = cv2.resize(attn, (bbox[2] - bbox[0], bbox[3] - bbox[1]), interpolation=cv2.INTER_LINEAR)
    full_heatmap = np.zeros((img_full.size[1], img_full.size[0]), dtype=np.float32)
    full_heatmap[bbox[1]:bbox[3], bbox[0]:bbox[2]] = res_heatmap
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img_full)
    plt.imshow(full_heatmap, cmap='jet', alpha=0.5)
    plt.axis("off")
    res_txt = "Correct" if gt == pr else "Incorrect"
    gt_txt = "Real" if gt == 0 else "Fake"
    pr_txt = "Real" if pr == 0 else "Fake"
    plt.title(f"Effort {strategy} Focus: {res_txt} (GT:{gt_txt}, Pred:{pr_txt})")
    
    save_name = f"focus_{strategy}_{suffix}.png"
    plt.savefig(os.path.join(VIS_DIR, save_name), dpi=200, bbox_inches='tight')
    plt.close()
    return save_name

# ── Evaluation ───────────────────────────────────────────────────────────────
def run_evaluation(strategy):
    print(f"\n📊 Evaluating Effort Model | Strategy: {strategy}")
    data_dir = SRC_ROOT if strategy == "full" else FACE_ROOT
    weights_path = os.path.join(WEIGHTS_DIR, f"effort_{strategy}.pth")
    if not os.path.exists(weights_path):
        print(f"  ❌ No weights found at {weights_path}. Skipping.")
        return

    # Load Model
    with open(CONFIG_PATH, "r") as f: config = yaml.safe_load(f)
    model = DETECTOR[config.get("detector_type", "effort")](config).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, weights_only=False, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    dataset = EffortEvalDataset(os.path.join(data_dir, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    all_preds, all_labels, all_probs, all_paths = [], [], [], []
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Eval"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            output = model({"image": imgs}, inference=True)
            logits = output["cls"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"  Accuracy: {acc*100:.2f}% | AUC: {auc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"], digits=4))

    # Per-Class Accuracy
    class_names = ["real", "fake"]
    cls_accs = []
    print("  Per-Class Accuracy:")
    print(f"  {'-'*35}")
    for idx, cls_name in enumerate(class_names):
        cls_mask    = (all_labels == idx)
        cls_correct = (all_preds[cls_mask] == idx).sum()
        cls_total   = cls_mask.sum()
        cls_acc     = cls_correct / cls_total if cls_total > 0 else 0.0
        cls_accs.append(cls_acc)
        print(f"    {cls_name:<12} : {cls_correct:>5} / {cls_total:>5}  →  {cls_acc*100:.2f}%")
    print(f"  {'-'*35}")

    # Save to results file
    results_file = os.path.join(VIS_DIR, "..", "effort_finetuned_results.txt")
    with open(results_file, "a") as f:
        f.write(f"Effort_Finetuned,{strategy},{acc:.4f},{auc:.4f},{cls_accs[0]:.4f},{cls_accs[1]:.4f}\n")
    print(f"  💾 Results appended to {results_file}")

    # Misclassification Report
    misclass_file = os.path.join(WEIGHTS_DIR, f"misclassifications_{strategy}.txt")
    with open(misclass_file, "w") as f:
        f.write("Path,GroundTruth,Predicted\n")
        mis_indices = []
        cor_indices = []
        for i in range(len(all_labels)):
            if all_preds[i] != all_labels[i]:
                f.write(f"{all_paths[i]},{all_labels[i]},{all_preds[i]}\n")
                mis_indices.append(i)
            else:
                cor_indices.append(i)
    print(f"  📝 Saved misclassification report to {misclass_file}")

    # Visualization
    # 2 correct (1 real, 1 fake if possible)
    # 2 incorrect (1 real misclassified, 1 fake misclassified if possible)
    vis_queue = []
    # Pick Correct
    for idx in cor_indices:
        if all_labels[idx] == 0: vis_queue.append(('correct_real', idx)); break
    for idx in cor_indices:
        if all_labels[idx] == 1: vis_queue.append(('correct_fake', idx)); break
    # Pick Incorrect
    for idx in mis_indices:
        if all_labels[idx] == 0: vis_queue.append(('wrong_real_as_fake', idx)); break
    for idx in mis_indices:
        if all_labels[idx] == 1: vis_queue.append(('wrong_fake_as_real', idx)); break

    for suffix, idx in vis_queue:
        visualize_focus(model, all_paths[idx], int(all_labels[idx]), int(all_preds[idx]), suffix, strategy, DEVICE)
    print(f"  ✨ Visualizations saved in {VIS_DIR}")

if __name__ == "__main__":
    # Initialize results file with header
    results_file = os.path.join(VIS_DIR, "..", "effort_finetuned_results.txt")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        f.write("Model,Strategy,Accuracy,AUC,Acc_real,Acc_fake\n")

    for strat in ["full", "face"]:
        run_evaluation(strat)
