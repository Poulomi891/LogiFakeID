import os
import sys
import cv2
import torch
import numpy as np
import pickle
import tempfile
import jpegio
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DT_ROOT = "/Path/to/your/data/Manipulated_ID/DocTamper"
import timm.models.layers
sys.modules['timm.models.layers.drop'] = timm.models.layers

sys.path.insert(0, os.path.join(DT_ROOT, "models"))
from dtd import seg_dtd, VPH, AddCoords, ConvBlock, LayerNorm, SCSEModule, ConvBNReLU, FUSE1, FUSE2, FUSE3, MID, DTD
from swins import BasicLayer, SwinTransformerBlock, WindowAttention, Mlp, PatchMerging, PatchEmbed, SwinTransformerV2

FANTASY_ROOT = "/Path/to/your/data/Manipulated_ID/FANTASYID_DATASET"
CHECKPOINT = os.path.join(DT_ROOT, "checkpoints/dtd_doctamper.pth")

class FantasyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        import pandas as pd
        csv_path = os.path.join(FANTASY_ROOT, "fantasyIDiap-test.csv")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rel_path = row['path']
            is_attack = str(row['is_attack']).lower() == 'true'
            label = 0 if is_attack else 1 # Fake=0, Real=1 for DocTamper
            abs_path = os.path.join(FANTASY_ROOT, rel_path)
            if os.path.exists(abs_path):
                self.samples.append((abs_path, label))
                    
        with open(os.path.join(DT_ROOT, 'qt_table.pk'), 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {k: torch.LongTensor(v) for k, v in pks.items()}
        self.default_q = 95
        self.toctsr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        from PIL import Image
        im = Image.open(path).convert('RGB')
        im = im.resize((512, 512), Image.BICUBIC)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            im.save(tmp.name, "JPEG", quality=self.default_q)
            jpg = jpegio.read(tmp.name)
            dct = jpg.coef_arrays[0].copy()
            dct = np.clip(np.abs(dct), 0, 20)
        
        return {
            'image': self.toctsr(im),
            'dct': torch.from_numpy(dct).long(),
            'qtb': self.pks[self.default_q].reshape(1, 8, 8),
            'label': label,
            'path': path
        }

def load_model():
    model = seg_dtd('', 2).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    for m in model.modules():
        if isinstance(m, torch.nn.GELU):
            if not hasattr(m, 'approximate'): m.approximate = 'none'
        if 'DropPath' in str(type(m)):
            if not hasattr(m, 'scale_by_keep'): m.scale_by_keep = True
    model.eval()
    return model

def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except: eer = fpr[np.nanargmin(np.absolute(((1-tpr) - fpr)))]
    return eer

def test_doctamper_fantasy():
    print(f"\n🚀 Evaluating DocTamper on Fantasy ID Dataset")
    dataset = FantasyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    model = load_model()
    all_scores, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            imgs = batch['image'].to(DEVICE)
            dcts = batch['dct'].to(DEVICE)
            qtbs = batch['qtb'].to(DEVICE)
            labels = batch['label'] # Fake=0, Real=1
            
            logits = model(imgs, dcts, qtbs)
            probs = F.softmax(logits, dim=1)
            
            tamper_probs = probs[:, 1, :, :]
            mask = tamper_probs > 0.3
            scores = (tamper_probs * mask).sum(dim=[1,2]) / mask.sum(dim=[1,2]).clamp(min=1)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # fake = 0, real = 1
    # score > 0.5 implies fake => pred 0
    all_preds = (all_scores <= 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, 1 - all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    
    # EER calculation (fake=0 as positive class -> 1 - labels)
    eer = compute_eer(1 - all_labels, all_scores)
    
    print(f"\n📊 Results: Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | EER: {eer:.4f}")

if __name__ == "__main__":
    test_doctamper_fantasy()
