# LogiFakeID: Document Forgery Detection

This repository contains the source code for the document forgery detection models and baseline evaluations.

## Requirements

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Structure

- Various model directories (e.g., `ADCD_Net`, `ASCFormer`, `DocTamper`, `Effort`, `FreqNet`, `IML-ViT`, `MMFusion_ML`, `ProDet`, `SAFIRE`, `TruFor`) containing evaluation scripts.
- `ocr_baseline_easyocr.py`: Script for OCR baseline evaluation.

## Data Preparation

Update the paths in the respective scripts. By default, paths are set to `/Path/to/your/data/...`. You should change this to the location where your dataset is stored.

## Running Evaluations

Each model directory contains its own test/eval scripts. Navigate to the desired model directory and execute the script.
For example:
```bash
cd DocTamper
python eval_doctamper.py
```
