import os
import sys
import re
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import warnings
from difflib import SequenceMatcher
warnings.filterwarnings("ignore")

# Attempt to import EasyOCR
try:
    import easyocr
    READER = easyocr.Reader(['en', 'hi'], gpu=True) # Will fallback to CPU if no GPU
except ImportError:
    READER = None
    print("❌ EasyOCR not found. Please install it with 'pip install easyocr'")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split"
RESULTS_DIR = "/Path/to/your/data/Manipulated_ID/Results/ocr_baseline"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Step 1: Symbolic Rule Validators ────────────────────────────────────────

# Verhoeff tables
D = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4], [2,7,0,5,6,3,9,8,1,4],
    [3,6,5,0,9,1,8,7,4,2], [4,2,6,9,0,7,1,5,3,8], [5,8,1,4,7,0,9,2,6,3],
    [6,3,8,1,5,9,4,2,7,0], [7,4,9,2,3,6,0,1,8,5], [8,9,3,7,1,4,5,6,0,2],
    [9,0,4,8,6,2,7,3,5,1]
]
P = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4], [5,8,0,3,7,9,6,1,4,2],
    [8,9,7,2,0,4,1,5,3,6], [9,4,0,5,7,2,8,1,3,6], [4,2,7,0,8,6,9,1,5,3],
    [2,6,0,8,9,1,4,5,3,7], [6,1,8,9,4,5,2,3,7,0], [1,3,9,4,2,0,6,7,5,8],
    [3,7,4,6,5,8,1,0,2,9]
]

def verhoeff_check(number: str) -> bool:
    """Returns True if number passes Verhoeff checksum."""
    if not number or not number.isdigit() or len(number) != 12:
        return False
    c = 0
    digits = [int(d) for d in reversed(number)]
    for i, digit in enumerate(digits):
        c = D[c][P[i % 8][digit]]
    return c == 0

def check_date_logic(fields: dict) -> dict:
    results = {'date_logic_ok': True}
    fmt = '%d/%m/%Y'
    try:
        dates = {}
        for key in ['dob', 'issue_date', 'details_date']:
            val = fields.get(key)
            if val:
                # Basic cleaning for OCR noise
                val = val.replace('.', '/').replace('-', '/')
                dates[key] = datetime.strptime(val, fmt)
        
        today = datetime.today()
        
        if 'dob' in dates and 'issue_date' in dates:
            results['dob_before_issue'] = dates['dob'] < dates['issue_date']
            if not results['dob_before_issue']: results['date_logic_ok'] = False
            
        if 'issue_date' in dates:
            results['issue_before_today'] = dates['issue_date'] <= today
            if not results['issue_before_today']: results['date_logic_ok'] = False
            
        if 'dob' in dates:
            age = (today - dates['dob']).days / 365
            results['age_valid'] = 0 < age < 120
            if not results['age_valid']: results['date_logic_ok'] = False

    except Exception:
        results['date_logic_ok'] = None # Could not parse
    return results

def fuzzy_match(text, target, threshold=0.7):
    if not text or not target: return False
    return SequenceMatcher(None, text.upper(), target.upper()).ratio() >= threshold

SPELLING_VARIANTS = {'ADHAR': 'AADHAAR', 'AADHAR': 'AADHAAR', 'Govenrment': 'Government', 'Goverment': 'Government'}

def check_template_strings(raw_text: str) -> dict:
    results = {}
    raw_upper = raw_text.upper()
    
    # Check for obvious spelling errors (not just OCR noise)
    results['has_spelling_error'] = any(k.upper() in raw_upper for k in SPELLING_VARIANTS.keys())
    
    # Use fuzzy matching for template strings
    results['has_correct_website'] = fuzzy_match(raw_upper, 'UIDAI.GOV.IN', 0.8) or 'UIDAI.GOV.IN' in raw_upper
    results['authority_string_ok'] = fuzzy_match(raw_upper, 'UNIQUE IDENTIFICATION AUTHORITY', 0.8) or 'UNIQUE IDENTIFICATION AUTHORITY' in raw_upper
    
    results['template_ok'] = (not results['has_spelling_error'] and 
                              (results['has_correct_website'] or results['authority_string_ok']))
    return results

# ── Step 2: Field Parsing ────────────────────────────────────────────────────

def parse_fields(ocr_text):
    fields = {'raw_text': ocr_text}
    
    # Clean OCR text slightly
    text = ocr_text.replace('\n', ' ')
    
    # Aadhaar number: 12 digits (usually 3 groups of 4)
    # Be more flexible with spaces and common OCR character errors
    text_clean = text.replace('O', '0').replace('I', '1').replace('l', '1').replace('|', '1').replace('S', '5')
    aadhaar = re.findall(r'\b(\d{4}\s*\d{4}\s*\d{4})\b', text_clean)
    if aadhaar:
        fields['aadhaar'] = aadhaar[0].replace(' ', '')
    else:
        # Fallback: find any 12 digit sequence
        any_12 = re.findall(r'\b(\d{12})\b', text_clean)
        if any_12:
            fields['aadhaar'] = any_12[0]
    
    # Dates: DD/MM/YYYY
    dates = re.findall(r'\b(\d{2}/\d{2}/\d{4})\b', text)
    if len(dates) >= 1:
        # Heuristic: oldest is DOB, newest is issue/details
        parsed_dates = []
        for d in dates:
            try: parsed_dates.append(datetime.strptime(d, '%d/%m/%Y'))
            except: pass
        if parsed_dates:
            parsed_dates.sort()
            fields['dob'] = parsed_dates[0].strftime('%d/%m/%Y')
            if len(parsed_dates) > 1:
                fields['issue_date'] = parsed_dates[-1].strftime('%d/%m/%Y')
    
    return fields

# ── Step 3: Full Pipeline ────────────────────────────────────────────────────

def predict_single(image_path, strategy="Full_ID"):
    if READER is None: return 0, [], 0.0
    
    # EasyOCR extraction
    try:
        results = READER.readtext(image_path)
        ocr_text = " ".join([res[1] for res in results])
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return 0, ["ocr_error"], 0.5

    fields = parse_fields(ocr_text)
    violations = []
    violation_score = 0.0
    
    # Rule 1: Verhoeff (Only if not Template_Only)
    if strategy != "Template_Only":
        if 'aadhaar' in fields:
            if not verhoeff_check(fields['aadhaar']):
                violations.append('invalid_aadhaar_checksum')
                violation_score += 0.4
        else:
            # Missing Aadhaar is a weak indicator because OCR might just fail
            violations.append('missing_aadhaar_number')
            violation_score += 0.2
    
    # Rule 2: Date logic (Only if not Template_Only)
    if strategy != "Template_Only":
        date_result = check_date_logic(fields)
        if date_result.get('date_logic_ok') == False:
            violations.append('invalid_date_logic')
            violation_score += 0.3
    
    # Rule 3: Template strings (Only if not Content_Only)
    if strategy != "Content_Only":
        template_result = check_template_strings(ocr_text)
        if not template_result['template_ok']:
            violations.append('template_violation')
            if template_result.get('has_spelling_error'):
                violation_score += 0.8 # Strong indicator
            else:
                violation_score += 0.4 # Missing template strings
    
    # Final prediction based on a threshold (can be tuned)
    # For AUC calculation, we use the score
    prediction = 1 if violation_score >= 0.4 else 0
    return prediction, violations, violation_score

# ── Step 4: Run Evaluation ───────────────────────────────────────────────────

def run_strategy_bench(strategy_name, data_root):
    print(f"\n📂 Benchmarking OCR Baseline on Strategy: {strategy_name}")
    results = []
    
    # Process Real and Fake folders
    for label, gt in [('real', 0), ('fake', 1)]:
        folder = os.path.join(data_root, 'test', label)
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue
        
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        # Process samples
        for fname in tqdm(files, desc=f"Processing {label}"):
            path = os.path.join(folder, fname)
            pred, violations, score = predict_single(path, strategy=strategy_name)
            results.append({
                'filename': fname,
                'gt': gt,
                'pred': pred,
                'score': score,
                'violations': "|".join(violations)
            })
    
    if not results: return
    
    df = pd.DataFrame(results)
    acc = accuracy_score(df['gt'], df['pred'])
    error_rate = 1 - acc
    
    # Use score for AUC if possible
    try:
        auc = roc_auc_score(df['gt'], df['score'])
    except:
        auc = accuracy_score(df['gt'], df['pred']) # Fallback
    
    print(f"\n📊 {strategy_name} Results:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Error Rate: {error_rate:.4f}")
    print(f"   AUC     : {auc:.4f}")
    print(classification_report(df['gt'], df['pred']))
    
    out_csv = os.path.join(RESULTS_DIR, f"ocr_results_{strategy_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"📝 Detailed results saved to: {out_csv}")

if __name__ == "__main__":
    strategies = {
        "Content_Only": "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_content",
        "Template_Only": "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split_template",
        "Full_ID": "/Path/to/your/data/Manipulated_ID/Final_ID_dataset_split"
    }
    
    for name, path in strategies.items():
        run_strategy_bench(name, path)
