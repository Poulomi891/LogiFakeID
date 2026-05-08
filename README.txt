# LogiFakeID: Document Forgery Generation and Benchmarking.

This repository contains the source code for the document forgery generation, benchmarking and evaluations.

For reference, we have added the dataset generation code in Aadhar_card_generation.ipynb

## Requirements for benchmarking and evaluation

Install the dependencies using:
```bash
pip install -r requirements.txt
```
Download the LogiFakeID dataset and place it in your preferred location.
Update the dataset root paths in the respective scripts. By default, paths are set to /Path/to/your/data/.... Replace these with the actual path to your dataset.
The expected dataset directory structure is: (after 70:30 splitting)
Final_ID_dataset_split/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/

Final_ID_dataset_split_faces/       # Face-cropped inputs (S2 - Face Only)
Final_ID_dataset_split_template/    # Template-masked inputs (S3 - Template Only)
Final_ID_dataset_split_content/     # Content-masked inputs (S4 - Content Only)
## Structure

Before running evaluations under the S3-Template Only and S4-Content Only strategies, you must generate the regionally masked inputs using:
python create_regional_mask.py

 All model checkpoints must be downloaded from their respective official repositories. Pre-trained weights are NOT included in this repository.

- Various model directories (e.g., `ADCD_Net`, `ASCFormer`, `DocTamper`, `Effort`, `FreqNet`, `IML-ViT`, `MMFusion_ML`, `ProDet`, `SAFIRE`, `TruFor`) containing evaluation scripts.

To evaluate a model across all strategies and obtain overall AUC, EER, accuracy, and per-class metrics, run:
cd <ModelDirectory>
python test_{model}_pretrained.py

To evaluate per-manipulation-type performance (Spelling Errors, Word Jumbling, GI Substitution, Invalid Date, Invalid ID, Gender Mismatch), run:
cd <ModelDirectory>
python eval_{model}.py

To run the training-free OCR-based rule-driven baseline: 
python ocr_baseline_easyocr.py

To evaluate models on the FantasyID dataset for cross-dataset comparison, run:
cd <ModelDirectory>
python eval_{model}_fantasy.py

'''
