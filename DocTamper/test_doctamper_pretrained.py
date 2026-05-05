"""
test_doctamper_pretrained.py — Test the pretrained DocTamper on ID dataset
  Strategies: Full ID, Face-Only, Template-Only, Content-Only
  Alignment: ImageFolder (fake=0, real=1).
  DocTamper outputs a segmentation mask (0=real, 1=tampered).
  We use the maximum probability of the tampered class as the image score.
"""
import os, sys, cv2, torch, numpy as np, pickle, tempfile, jpegio
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Setup paths
DT_ROOT = "/Path/to/your/data/Manipulated_ID/DocTamper"
import timm.models.layers
sys.modules['timm.models.layers.drop'] = timm.models.layers

sys.path.insert(0, os.path.join(DT_ROOT, "models"))
from dtd import seg_dtd, VPH, AddCoords, ConvBlock, LayerNorm, SCSEModule, ConvBNReLU, FUSE1, FUSE2, FUSE3, MID, DTD
from swins import BasicLayer, SwinTransformerBlock, WindowAttention, Mlp, PatchMerging, PatchEmbed, SwinTransformerV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.path.join(DT_ROOT, "checkpoints/dtd_doctamper.pth")
RESULTS_DIR = "/Path/to/your/data/Manipulated_ID/Results/results_doctamper_pretrained"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset paths
SRC_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split"
FACE_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_faces"
TEMPLATE_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template"
CONTENT_ROOT = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content"

# Ensure support files are present in CWD for DTD model
for f in ['vph_imagenet.pt', 'swin_imagenet.pt']:
    src = os.path.join(DT_ROOT, 'checkpoints', f)
    if not os.path.exists(f): 
        try: os.symlink(src, f)
        except: pass

class DocTamperInferenceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = datasets.ImageFolder(data_dir)
        self.transform = transform
        with open(os.path.join(DT_ROOT, 'qt_table.pk'), 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {k: torch.LongTensor(v) for k, v in pks.items()}
        # Use a default quality for JPEG extraction if needed
        self.default_q = 95
        self.toctsr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self): return len(self.dataset)

    def __getitem__(self, index):
        path, label = self.dataset.samples[index]
        from PIL import Image
        im = Image.open(path).convert('RGB')
        
        # DocTamper needs specific size? eval_dtd doesn't resize, but uses fixed size buckets or similar.
        # However, for benchmarking, we usually resize to a fixed size. 
        # The model seems to handle various sizes but let's use 512x512 or similar if common.
        # ProDet used 256. FreqNet used 224. 
        # DocTamper LMDBs are often 256 or 512. Let's use 512 to be safe for document details.
        im = im.resize((512, 512), Image.BICUBIC)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            im.save(tmp.name, "JPEG", quality=self.default_q)
            jpg = jpegio.read(tmp.name)
            dct = jpg.coef_arrays[0].copy()
            # The model expects dct to be absolute and clipped
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
    # model = torch.nn.DataParallel(model)  # Explicitly disabled as requested
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['state_dict']
    
    # Strip 'module.' prefix from state_dict keys if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded checkpoint. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    if len(missing) > 0: print(f"    ⚠️ Missing keys: {missing[:5]}...")
    if len(unexpected) > 0: print(f"    ⚠️ Unexpected keys: {unexpected[:5]}...")
    
    # Fix for GELU and DropPath version mismatch
    for m in model.modules():
        if isinstance(m, torch.nn.GELU):
            if not hasattr(m, 'approximate'):
                m.approximate = 'none'
        if 'DropPath' in str(type(m)):
            if not hasattr(m, 'scale_by_keep'):
                m.scale_by_keep = True

    model.eval()
    return model

def test_strategy(strategy):
    print(f"\n🚀 Testing PRETRAINED DocTamper | Strategy: {strategy}")
    if strategy == "full": data_dir = os.path.join(SRC_ROOT, 'test')
    elif strategy == "face": data_dir = os.path.join(FACE_ROOT, 'test')
    elif strategy == "template": data_dir = os.path.join(TEMPLATE_ROOT, 'test')
    elif strategy == "content": data_dir = os.path.join(CONTENT_ROOT, 'test')
    else: raise ValueError(f"Unknown strategy: {strategy}")
    
    dataset = DocTamperInferenceDataset(data_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"  Samples: {len(dataset)}")

    model = load_model()
    all_scores, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            imgs = batch['image'].to(DEVICE)
            dcts = batch['dct'].to(DEVICE)
            qtbs = batch['qtb'].to(DEVICE)
            labels = batch['label']
            
            # Predict
            logits = model(imgs, dcts, qtbs) # [B, 2, H, W]
            probs = F.softmax(logits, dim=1) # [B, 2, H, W]
            
            # Tamper score = max probability of tampered class (1) across the mask
            tamper_probs = probs[:, 1, :, :] # [B, H, W]
            mask = tamper_probs > 0.3
            scores = (tamper_probs * mask).sum(dim=[1,2]) / mask.sum(dim=[1,2]).clamp(min=1)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 1. Check raw score distribution BEFORE thresholding
    print(f"\n📊 Diagnostics for {strategy}:")
    print(f"  Score stats: min={all_scores.min():.4f}, max={all_scores.max():.4f}, mean={all_scores.mean():.4f}, std={all_scores.std():.4f}")
    
    # 2. Check if scores are degenerate (all same value)
    print(f"  Unique score values: {len(np.unique(all_scores.round(3)))}")
    
    # 3. Check class balance in dataset
    try:
        print(f"  Class mapping: {loader.dataset.dataset.class_to_idx}")
        from collections import Counter
        print(f"  Sample counts: {Counter(l for _,l in loader.dataset.dataset.samples)}")
    except: pass

    # 4. Visualize per-class score distributions
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for cls, name in [(0,'fake'),(1,'real')]:
            m = all_labels == cls
            if np.any(m):
                plt.hist(all_scores[m], bins=50, alpha=0.5, label=name)
        plt.title(f"Score Distribution - {strategy}")
        plt.xlabel("Tamper Probability (High = Fake)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"score_dist_{strategy}.png"))
        plt.close()
        print(f"  📈 Histogram saved to: {RESULTS_DIR}/score_dist_{strategy}.png")
    except Exception as e:
        print(f"  ⚠️ Could not save histogram: {e}")

    # Process metrics
    # NOTE: Our labels are fake=0, real=1. 
    # If score > 0.5 (DocTamper says tampered), prediction should be 0 (fake).
    # If score <= 0.5 (DocTamper says real), prediction should be 1 (real).
    all_preds = (all_scores <= 0.5).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    # AUC score for label 1 (real). So prob of real = 1 - all_scores
    auc = roc_auc_score(all_labels, 1 - all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    
    if len(np.unique(all_labels)) > 1:
        # Treat fake (0) as positive class for EER calculation
        fpr_curve, tpr_curve, thresholds = roc_curve(1 - all_labels, all_scores)
        eer = brentq(lambda x : 1. - x - interp1d(fpr_curve, tpr_curve)(x), 0., 1.)
    else:
        eer = 0.0

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tp_fake = cm[0, 0]
    fn_fake = cm[0, 1]
    fp_fake = cm[1, 0]
    tn_fake = cm[1, 1]
    
    fpr = (fp_fake / (fp_fake + tn_fake) * 100) if (fp_fake + tn_fake) > 0 else 0.0
    fnr = (fn_fake / (fn_fake + tp_fake) * 100) if (fn_fake + tp_fake) > 0 else 0.0
    
    classes = ['fake', 'real']
    print(f"\n📈 DocTamper Pretrained {strategy}: Acc={acc*100:.2f}% | AUC={auc:.4f} | EER={eer:.4f}")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
    ca = []
    for i, c in enumerate(classes):
        m = all_labels == i
        ca.append((all_preds[m] == i).sum() / m.sum() if m.sum() > 0 else 0)
        print(f"    {c:<12}: {(all_preds[m]==i).sum():>5}/{m.sum():>5} → {ca[-1]*100:.2f}%")
        
    print(f"    FPR         : {fpr:.2f}%")
    print(f"    FNR         : {fnr:.2f}%")
        
    with open(os.path.join(RESULTS_DIR, "doctamper_pretrained_results.txt"), "a") as f:
        f.write(f"DocTamper_Pretrained,{strategy},{acc:.4f},{auc:.4f},{eer:.4f},{ca[0]:.4f},{ca[1]:.4f},{fpr:.4f},{fnr:.4f}\n")

if __name__ == "__main__":
    with open(os.path.join(RESULTS_DIR, "doctamper_pretrained_results.txt"), "w") as f:
        f.write("Model,Strategy,Accuracy,AUC,EER,Acc_fake,Acc_real,FPR,FNR\n")
    for s in ["full", "face", "template", "content"]: test_strategy(s)
    print("\n✅ DocTamper pretrained testing complete!")
