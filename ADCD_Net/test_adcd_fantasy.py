import os
import sys
import pickle
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# --- NumPy 2.0 Pickle Compatibility Fix ---
class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.core.numeric": module = "numpy._core.numeric"
        elif module == "numpy.core.multiarray": module = "numpy._core.multiarray"
        elif module == "numpy.core.umath": module = "numpy._core.umath"
        if module not in sys.modules:
            try: __import__(module)
            except ImportError:
                fallback = module.replace("numpy._core", "numpy.core")
                try: __import__(fallback); module = fallback
                except ImportError: pass
        return super().find_class(module, name)

def load_qt(qt_path):
    try: import numpy._core.numeric; import numpy._core.multiarray; import numpy._core.umath
    except ImportError: pass
    try: import numpy.core.numeric; import numpy.core.multiarray; import numpy.core.umath
    except ImportError: pass

    with open(qt_path, 'rb') as f:
        try: pks_ = CompatibilityUnpickler(f).load()
        except Exception: f.seek(0); pks_ = pickle.load(f)
    return {k: torch.LongTensor(v) for k, v in pks_.items()}

# ───────────────── Configuration ──────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_MAX_SIZE = 512
IMG_MEAN = (0.485, 0.455, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

REPO_ROOT = "/Path/to/your/data/Manipulated_ID/ADCD-Net"
sys.path.insert(0, REPO_ROOT)

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
DOCRES_PATH = os.path.join(REPO_ROOT, "ADCD-Net_exp_data", "docres.pkl")
CKPT_PATH   = os.path.join(REPO_ROOT, "ADCD-Net_exp_data",  "ADCDNet.pth")
QT_PATH     = os.path.join(REPO_ROOT, "ADCD-Net_exp_data", "qt_table.pk")

def img_to_tensor(img_pil):
    arr = np.array(img_pil).astype(np.float32) / 255.0
    arr = (arr - np.array(IMG_MEAN)) / np.array(IMG_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()

def extract_dct(img_pil):
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
    if t.dim() == 3: _, h, w = t.shape
    else: h, w = t.shape
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    new_s = max(new_h, new_w)
    pad_h, pad_w = new_s - h, new_s - w
    if t.dim() == 3: return F.pad(t, (0, pad_w, 0, pad_h), value=0.0), h, w
    else: return F.pad(t, (0, pad_w, 0, pad_h), value=0), h, w

def resize_if_needed(img_cv2, max_size):
    h, w = img_cv2.shape[:2]
    if max(h, w) <= max_size: return img_cv2
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

@torch.no_grad()
def predict_score(model, img_path, qts):
    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None: return 0.0
    img_cv2 = resize_if_needed(img_cv2, VAL_MAX_SIZE)
    h, w = img_cv2.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    dct, img_pil_reopened = extract_dct(img_pil)
    qt = qts[100]

    img_t = img_to_tensor(img_pil_reopened)
    ocr_mask = torch.zeros(1, h, w, dtype=torch.long)
    mask = torch.zeros(1, h, w, dtype=torch.long)

    img_t, orig_h, orig_w = pad_to_multiple(img_t, 16)
    dct_t = torch.tensor(np.clip(np.abs(dct), 0, 20), dtype=torch.long)
    dct_t, _, _ = pad_to_multiple(dct_t, 16)
    mask, _, _ = pad_to_multiple(mask.squeeze(0), 16)
    mask = mask.unsqueeze(0)
    ocr_mask, _, _ = pad_to_multiple(ocr_mask.squeeze(0), 16)
    ocr_mask = ocr_mask.unsqueeze(0)

    img_t = img_t.unsqueeze(0).to(DEVICE)
    dct_t = dct_t.unsqueeze(0).long().to(DEVICE)
    qt_t = qt.unsqueeze(0).to(DEVICE)
    mask = mask.unsqueeze(0).to(DEVICE)
    ocr_mask = ocr_mask.unsqueeze(0).to(DEVICE)

    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(img_t, dct_t, qt_t, mask, ocr_mask, is_train=False)[0]

    prob_map = F.softmax(logits, dim=1)[0, 1, :orig_h, :orig_w]
    return float(prob_map.float().cpu().numpy().max())

def test_adcd_fantasy():
    print(f"\n🚀 Evaluating ADCD-Net on Fantasy ID Dataset")
    qts = load_qt(QT_PATH)
    
    import cfg
    cfg.docres_ckpt_path = DOCRES_PATH
    from model.model import ADCDNet
    
    model = ADCDNet()
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    state = ckpt['model']
    clean = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    model.to(DEVICE).eval()

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
        score = predict_score(model, path, qts)
        all_scores.append(score)
        all_labels.append(label)

    all_scores, all_labels = np.array(all_scores), np.array(all_labels)
    all_preds = (all_scores > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    eer = compute_eer(all_labels, all_scores)
    
    print(f"\n📊 Results: Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")

if __name__ == "__main__":
    test_adcd_fantasy()
