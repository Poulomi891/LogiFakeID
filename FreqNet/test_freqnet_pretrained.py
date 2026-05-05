"""
test_freqnet_pretrained.py — Test the pretrained FreqNet directly on ID dataset
  Strategies: Full ID, Face-Only, Template-Only, Content-Only (no fine-tuning)
  FreqNet outputs 1 logit by default (BCE). We use sigmoid > 0.5 for classification.
  Convention: label 0 = fake, label 1 = real (matching ImageFolder alphabetical order: fake=0, real=1)
  FreqNet was trained with: output > 0.5 = fake, < 0.5 = real
  Alignment: ImageFolder maps fake->0, real->1. Model sigmoid > 0.5 (fake) -> pred 0.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np, torch, torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/Path/to/your/data/Manipulated_ID/FreqNet-DeepfakeDetection")
from networks.freqnet import FreqNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split"
FACE_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_faces"
TEMPLATE_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template"
CONTENT_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content"
CHECKPOINT = "/Path/to/your/data/Manipulated_ID/FreqNet-DeepfakeDetection/checkpoints/pretrained_freqnet.pth"
RESULTS_DIR = "/Path/to/your/data/Manipulated_ID/Results/results_freqnet_pretrained"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def load_model():
    model = FreqNet(num_classes=1)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model.to(DEVICE)

def test_strategy(strategy):
    print(f"\n🚀 Testing PRETRAINED FreqNet | Strategy: {strategy}")
    if strategy == "full": data_dir = SRC_ROOT
    elif strategy == "face": data_dir = FACE_ROOT
    elif strategy == "template": data_dir = TEMPLATE_ROOT
    elif strategy == "content": data_dir = CONTENT_ROOT
    else: raise ValueError(f"Unknown strategy: {strategy}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    print(f"  Classes: {dataset.classes} | Samples: {len(dataset)}")

    model = load_model()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs).squeeze(1)  # [B] single logit
            prob_fake = torch.sigmoid(logits)
            # ImageFolder: fake=0, real=1 alphabetically
            # FreqNet: sigmoid > 0.5 = fake (cls 0 in ImageFolder)
            preds = (prob_fake < 0.5).long()  # invert: low prob = real (cls 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend((1 - prob_fake).cpu().numpy()) # prob_real for metrics

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    print(f"\n📈 FreqNet Pretrained {strategy}: Acc={acc*100:.2f}% | AUC={auc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes, digits=4))
    ca = []
    for i, c in enumerate(dataset.classes):
        m = all_labels == i; ca.append((all_preds[m] == i).sum() / m.sum() if m.sum() > 0 else 0)
        print(f"    {c:<12}: {(all_preds[m]==i).sum():>5}/{m.sum():>5} → {ca[-1]*100:.2f}%")
    
    # Calculate advanced metrics
    eer = compute_eer(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # real as fake
    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # fake as real
    fpr = apcer 
    fnr = bpcer
    
    print(f"    EER         : {eer:.4f}")
    print(f"    BPCER (FPR) : {bpcer:.4f}")
    print(f"    APCER (FNR) : {apcer:.4f}")
    print(f"    FPR         : {fpr:.4f}")
    print(f"    FNR         : {fnr:.4f}")

    with open(os.path.join(RESULTS_DIR, "freqnet_pretrained_results.txt"), "a") as f:
        f.write(f"FreqNet_Pretrained,{strategy},{acc:.4f},{auc:.4f},{ca[0]:.4f},{ca[1]:.4f},{eer:.4f},{bpcer:.4f},{apcer:.4f},{fpr:.4f},{fnr:.4f}\n")

    # Misclassification report
    mf = os.path.join(RESULTS_DIR, f"misclass_{strategy}.txt")
    with open(mf, "w") as f:
        f.write("Path,GT,Pred\n")
        for i in range(len(all_labels)):
            if all_preds[i] != all_labels[i]:
                f.write(f"{dataset.samples[i][0]},{all_labels[i]},{all_preds[i]}\n")
    print(f"  📝 Misclassification report: {mf}")

if __name__ == "__main__":
    with open(os.path.join(RESULTS_DIR, "freqnet_pretrained_results.txt"), "w") as f:
        f.write("Model,Strategy,Accuracy,AUC,Acc_fake,Acc_real,EER,BPCER,APCER,FPR,FNR\n")
    for s in ["full", "face", "template", "content"]: test_strategy(s)
    print("\n✅ FreqNet pretrained testing complete!")
