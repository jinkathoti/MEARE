# MEARE — Reproducibility Package

Code, data pointers, split indices, trained checkpoint, per-record predictions and result
tables for the paper *"MEARE: An Explainable Two-Branch Framework for Acute Myocardial
Infarction Assessment from Cardiac Biomarkers and ECG Images, Evaluated on Separate Public
Cohorts"* (VFAST Transactions on Software Engineering).

All results in the paper are reproduced by the scripts below. No result was hand-edited.

## Datasets
- **Biomarker (tabular)**: `data/biomarker_heart_attack.csv` — Maghdid & Rashid, Mendeley Data v2,
  DOI 10.17632/65gxgy2nmg.2 (CC BY 4.0). SHA-256 of the analysed file:
  `f090acd9aa6ed0df3abd55832a82bf932dcf71987175f68617dd57fb074f4665`.
- **ECG**: PTB-XL v1.0.3 from PhysioNet (https://physionet.org/content/ptb-xl/).
  `data/ptbxl_mi_norm_labels_and_folds.csv` gives the exact MI-vs-NORM label and the official
  patient-disjoint `strat_fold` for every `ecg_id` used (15,013 records; 5,486 MI, 9,527 NORM).
  MI/NORM selection: SCP statements mapped to diagnostic superclasses (scp_statements.csv,
  diagnostic=1); MI-positive if superclasses contain MI, MI-negative if they contain NORM but
  not MI, excluded otherwise; MI takes precedence on overlap.

## Environment
Python 3.12, scikit-learn 1.8, PyTorch 2.x (CPU is sufficient for the tabular branch and
evaluation; a GPU speeds up ECG training). `pip install numpy pandas scikit-learn scipy torch
torchvision matplotlib wfdb`.

## Splits and seeds
- Tabular: single stratified split, `test_size=0.2`, `random_state=42` (training 1,055, test 264);
  5-fold stratified CV (seed 42) on the training set only; operating threshold = Youden index of
  the out-of-fold training probabilities (0.65), applied once to the test set.
- ECG: official PTB-XL folds 1–8 train, 9 validation, 10 test (patient-disjoint).
- Controlled comparison: balanced 2,000-record subset, 3 seeds (0,1,2).

## Reproduce
| Result | Script | Output |
|---|---|---|
| Tabular CV + held-out + ablation + subgroups | `code/meare_model.py` | metrics, ROC/PR/importance |
| Separability + incremental value (Tables 6–7) | `code/tab_incremental.py` | `results/tab_incremental.json` |
| Calibration (ECE) + reliability figure; patient-level bootstrap (Table 10) | `code/calib_patient.py` | `results/calibration.json`, `results/patient_bootstrap.json`, `fig_calib.png` |
| ECG image checkpoint eval + raw-signal baselines (Table 9) | `code/ecg_pipeline.py`, `code/ecg_raw_baseline.py` | per-record predictions |
| Controlled image-vs-raw, 3 seeds (Table 10) | `code/ecg_controlled.py` | `results/ecg_ctrl_*_s*.csv`, `results/ecg_controlled.json` |
| Grad-CAM territory + deletion test | `code/gradcam_eval.py` | `results/gradcam_eval.json`, `results/gradcam_records.csv` |
| Reference verification | `code/verify_refs.py` | `results/ref_check.json` |

## Checkpoint and predictions
- `checkpoints/ecg_resnet18.pt` — deployed ECG-image ResNet-18 checkpoint.
- `results/ecg_ctrl_{image,raw}_s{0,1,2}.csv` — per-record test predictions (ecg_id, label, p).
- `results/gradcam_records.csv` — per-record Grad-CAM territory/deletion measures.

## Key reproduced numbers
- Tabular RF held-out: ROC-AUC 0.997, accuracy 98.1% (TP159/FN3/TN100/FP2), Brier 0.017, ECE 0.042.
- Two-threshold troponin/CK-MB rule: reproduces 98.6% of all labels; identical test decisions to RF (McNemar p=1.00).
- ECG image (deployed, full test fold): AUC 0.925; raw-signal full-data: 0.969.
- Controlled (2,000-record, 3 seeds): image 0.939±0.003 vs raw 0.940±0.002; paired diff −0.003
  (record-level p=0.567; patient-level cluster bootstrap p=0.61).
- Grad-CAM: lead-level territory association p<0.001; pixel deletion test unfavourable (Wilcoxon p<1e-7).

## Note
This is a research prototype evaluated on two separate public cohorts; it is not a validated
diagnostic tool and must not be used for clinical decisions.
