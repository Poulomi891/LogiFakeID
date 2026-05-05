import os
import sys
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import warnings

# ── Dynamic fix for libstdc++ version mismatch ───────────────────────────────
CONDA_LIB = "/home/tbvl_lab/miniconda3/envs/id_env/lib" # Adjust if TruFor env is different
if CONDA_LIB not in os.environ.get("LD_LIBRARY_PATH", "") and "RE_EXECED" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = CONDA_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["RE_EXECED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

warnings.filterwarnings("ignore")

# ── Environment Setup ────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Robust path handling: use script's directory if it contains models/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, "models")):
    MMFUSION_ROOT = SCRIPT_DIR
else:
    MMFUSION_ROOT = "/Path/to/your/data/Manipulated_ID/MMFusion-IML"

CKPT_PATH = os.path.join(MMFUSION_ROOT, "ckpt", "early_fusion_detection.pth")
RESULTS_FILE = os.path.join(MMFUSION_ROOT, "results_mmfusion_pretrained.txt")

if MMFUSION_ROOT not in sys.path:
    sys.path.insert(0, MMFUSION_ROOT)

try:
    from models.cmnext_conf import CMNeXtWithConf
    from models.modal_extract import ModalitiesExtractor
    from configs.cmnext_init_cfg import _C as config
except ImportError as e:
    print(f"❌ Error: Could not import MMFusion modules: {e}")
    print(f"   Checked MMFUSION_ROOT: {MMFUSION_ROOT}")
    print(f"   Current sys.path: {sys.path}")
    sys.exit(1)

# ── Dataset Definition ───────────────────────────────────────────────────────
class SimpleDataset(Dataset):
    def __init__(self, root, img_size=512):
        self.root = root
        self.img_size = img_size
        self.samples = []
        for label_name, label_val in [("real", 0), ("fake", 1)]:
            folder = os.path.join(root, label_name)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(folder, f), label_val))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_t = transforms.ToTensor()(img)
        return img_t, label

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

def evaluate_mmfusion():
    # ── Model Setup ──────────────────────────────────────────────────────────
    # Initialize with default config from early_fusion_detection
    config.defrost()
    config.MODEL.MODALS = ['img', 'noiseprint', 'bayar', 'srm']
    config.MODEL.BACKBONE = 'MixCMNeXtMHSA-B2'
    config.MODEL.DETECTION = 'confpool'
    config.MODEL.TRAIN_PHASE = 'detection'
    config.freeze()

    print(f"📦 Loading MMFusion from {CKPT_PATH}...")
    modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], None) # No np weights file needed, loading from ckpt
    model = CMNeXtWithConf(config.MODEL)
    
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    modal_extractor.load_state_dict(ckpt['extractor_state_dict'])
    
    modal_extractor = modal_extractor.to(DEVICE).eval()
    model = model.to(DEVICE).eval()

    STRATEGIES = {
        "Full_ID":          "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split",
        "Face_Crops":       "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_faces",
        "Template_Visible": "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content",
        "Content_Visible":  "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template"
    }

    with open(RESULTS_FILE, "w") as f_res:
        f_res.write("Strategy,Accuracy,AUC,Acc_Real,Acc_Fake,EER,BPCER,APCER,FPR,FNR\n")
        
        for name, root in STRATEGIES.items():
            test_path = os.path.join(root, "test")
            print(f"\n🚀 Evaluating Strategy: {name}")
            dataset = SimpleDataset(test_path, img_size=512)
            if len(dataset) == 0:
                print(f"   ⚠️ No data found for {name}")
                continue
                
            loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
            
            all_scores, all_labels = [], []
            for imgs, labels in tqdm(loader, desc=f"   {name}"):
                imgs = imgs.to(DEVICE)
                with torch.no_grad():
                    modals = modal_extractor(imgs)
                    imgs_norm = TF.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    inp = [imgs_norm] + modals
                    _, _, detection = model(inp)
                    scores = torch.sigmoid(detection).cpu().numpy().flatten()
                    all_scores.extend(scores)
                    all_labels.extend(labels.numpy())

            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)
            all_preds = (all_scores > 0.5).astype(int)
            
            acc = accuracy_score(all_labels, all_preds)
            try: auc = roc_auc_score(all_labels, all_scores)
            except: auc = 0.0
            
            # Per-class accuracy
            acc_real = np.mean(all_preds[all_labels == 0] == 0) if any(all_labels == 0) else 0.0
            acc_fake = np.mean(all_preds[all_labels == 1] == 1) if any(all_labels == 1) else 0.0
            
            # Calculate advanced metrics
            eer = compute_eer(all_labels, all_scores)
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
            # labels: [real(0), fake(1)]
            tn, fp, fn, tp = cm.ravel()
            bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # real as fake
            apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # fake as real
            fpr = bpcer
            fnr = apcer
            
            print(f"   ✅ Overall Acc: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")
            print(f"   ✅ Per-Class: Real {acc_real*100:.2f}%, Fake {acc_fake*100:.2f}%")
            print(f"   ✅ FPR (BPCER): {fpr:.4f}, FNR (APCER): {fnr:.4f}")
            
            f_res.write(f"{name},{acc:.4f},{auc:.4f},{acc_real:.4f},{acc_fake:.4f},{eer:.4f},{bpcer:.4f},{apcer:.4f},{fpr:.4f},{fnr:.4f}\n")

if __name__ == "__main__":
    evaluate_mmfusion()
