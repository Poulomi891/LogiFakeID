import os
import cv2
import numpy as np
import sys

# Ensure dlib can find its libs
CONDA_LIB = '/home/tbvl_lab/miniconda3/envs/reface_env/lib'
if CONDA_LIB not in os.environ.get('LD_LIBRARY_PATH', '') and 'RE_EXECED' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = CONDA_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['RE_EXECED'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)
import dlib

SRC_ROOT = "/DATA1/Poulomi/Manipulated_ID/Final_ID_dataset_split"
OUT_CONTENT = "/DATA1/Poulomi/Manipulated_ID/Final_ID_dataset_split_content"
OUT_TEMPLATE = "/DATA1/Poulomi/Manipulated_ID/Final_ID_dataset_split_template"

# ── PERFECTED BLACKOUT COORDINATES (Final Fine-Tuning v2) ────────────────────

# Style 1: Content Masked (Confirmed Perfect)
STYLE1_BLACKOUT = (0.00, 0.180, 1.00, 0.930) 

# Style 2: Template Masked (Content visible)
# Extended leftwards and kept vertical.
STYLE2_TEMPLATE_BOXES = [
    (0.015, 0.00, 1.000, 0.180), # Continuous Header Bar (Covers left logo to right edge)
    (0.000, 0.925, 1.000, 1.000), # Lower footer bar
]

# Tightened QR Box
QR_BOX = (0.790, 0.190, 0.990, 0.770)

# Face Area Fallback
FACE_HINT = (0.030, 0.150, 0.160, 0.680)

det = dlib.get_frontal_face_detector()

def process_image(src_path, content_path, template_path):
    img = cv2.imread(src_path)
    if img is None: return
    H, W = img.shape[:2]

    def draw_box(target, box, color=(0,0,0)):
        x1, y1, x2, y2 = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
        cv2.rectangle(target, (x1, y1), (x2, y2), color, -1)

    # --- STYLE 1: CONTENT MASKED ---
    img1 = img.copy()
    draw_box(img1, STYLE1_BLACKOUT)
    
    # --- STYLE 2: TEMPLATE MASKED ---
    img2 = img.copy()
    for box in STYLE2_TEMPLATE_BOXES:
        draw_box(img2, box)
    draw_box(img2, QR_BOX)
    
    # Precise Face Detection (Aggressive top/bottom margins)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = det(gray, 0)
    if rects:
        r = max(rects, key=lambda r: r.area())
        w, h = r.width(), r.height()
        # Cover whole photo area: 70% top margin (hair), 35% sides, 50% bottom (neck)
        fx1, fy1, fx2, fy2 = max(0,int(r.left()-0.35*w)), max(0,int(r.top()-0.70*h)), min(W,int(r.right()+0.35*w)), min(H,int(r.bottom()+0.50*h))
        cv2.rectangle(img2, (fx1, fy1), (fx2, fy2), (0,0,0), -1)
    else:
        draw_box(img2, FACE_HINT)

    cv2.imwrite(content_path, img1)
    cv2.imwrite(template_path, img2)

def run_masks(max_samples=None):
    count = 0
    for split in ["train", "test"]:
        for cls in ["fake", "real"]:
            src_dir = os.path.join(SRC_ROOT, split, cls)
            if not os.path.exists(src_dir): continue
            
            out_c_dir = os.path.join(OUT_CONTENT, split, cls)
            out_t_dir = os.path.join(OUT_TEMPLATE, split, cls)
            os.makedirs(out_c_dir, exist_ok=True)
            os.makedirs(out_t_dir, exist_ok=True)

            files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if max_samples: files = files[:max_samples]

            for fname in files:
                process_image(
                    os.path.join(src_dir, fname),
                    os.path.join(out_c_dir, fname),
                    os.path.join(out_t_dir, fname)
                )
                count += 1
                if count % 100 == 0: print(f"Processed {count} images...")
    print(f"Finished! Total images: {count}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running in TEST mode (2 images per folder)")
        run_masks(max_samples=2)
    else:
        run_masks()
