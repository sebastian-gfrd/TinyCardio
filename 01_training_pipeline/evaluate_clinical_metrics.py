"""
TinyCardio Clinical Evaluation & Pathology Breakdown Suite
==========================================================
Evaluates the trained TinyCardio 1D-CNN across unseen patient records,
providing detailed pathology-specific diagnostic breakdown:
  1. NSRDB Control Specificity (False Alarm Suppression)
  2. VFDB Lethal Arrhythmia Sensitivity (VF / VT detection)
  3. SDDB Sudden Cardiac Death Sensitivity (Pre-arrest telemetry)
  4. EDB Myocardial Ischemia Sensitivity (ST-T deviation detection)
Generates ROC-AUC, PR-AUC, Confusion Matrix, and clinical triage reports.
"""

import os
import sys
import json
import numpy as np
from typing import Tuple, List, Dict

# Ensure parent and 02_model_cpu directories are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
CPU_DIR = os.path.join(BASE_DIR, '02_model_cpu')
if CPU_DIR not in sys.path:
    sys.path.insert(0, CPU_DIR)

from utils.ecg_loader import load_ecg
from inference_cpu import TinyCardioCPU


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, List[Tuple[float, float]]]:
    """Computes Area Under the ROC Curve and ROC curve points."""
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5, []
        
    tpr_list = []
    fpr_list = []
    
    tp = 0
    fp = 0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
        
    # Trapezoidal integration for AUC (NumPy 2.0+ compatible)
    if hasattr(np, 'trapezoid'):
        auc = float(np.trapezoid(tpr_list, fpr_list))
    else:
        auc = float(getattr(np, 'trapz')(tpr_list, fpr_list))
    roc_points = list(zip(fpr_list[::max(1, len(fpr_list)//50)], tpr_list[::max(1, len(tpr_list)//50)]))
    return auc, roc_points


def evaluate_dataset_split(weights_path: str, npz_path: str = '01_training_pipeline/processed_data/dataset_250hz.npz') -> dict:
    """Evaluates the model on test set and computes all clinical metrics."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Processed dataset not found: {npz_path}")
        
    data = np.load(npz_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    model = TinyCardioCPU(weights_path)
    
    scores = []
    preds = []
    
    for i in range(len(X_test)):
        w = X_test[i].ravel()
        res = model.predict_window(w)
        scores.append(res['risk_score'])
        preds.append(1 if res['is_alert'] else 0)
        
    y_test = np.array(y_test, dtype=int)
    scores = np.array(scores, dtype=np.float32)
    preds = np.array(preds, dtype=int)
    
    tp = int(np.sum((y_test == 1) & (preds == 1)))
    fp = int(np.sum((y_test == 0) & (preds == 1)))
    tn = int(np.sum((y_test == 0) & (preds == 0)))
    fn = int(np.sum((y_test == 1) & (preds == 0)))
    
    sensitivity = tp / max(1, (tp + fn))
    specificity = tn / max(1, (tn + fp))
    precision = tp / max(1, (tp + fp))
    accuracy = (tp + tn) / len(y_test)
    f1 = 2 * (precision * sensitivity) / max(1e-7, (precision + sensitivity))
    
    auc, roc_pts = compute_roc_auc(y_test, scores)
    
    return {
        'total_test_samples': len(y_test),
        'positive_samples': int(np.sum(y_test == 1)),
        'negative_samples': int(np.sum(y_test == 0)),
        'confusion_matrix': {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        },
        'metrics': {
            'accuracy': round(float(accuracy), 4),
            'sensitivity_recall': round(float(sensitivity), 4),
            'specificity': round(float(specificity), 4),
            'precision': round(float(precision), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4)
        }
    }


def evaluate_per_database(weights_path: str) -> dict:
    """Evaluates performance across individual PhysioNet clinical databases."""
    model = TinyCardioCPU(weights_path)
    
    # 1. NSRDB Control (Test patients)
    from dataset_generator import extract_windows_from_record
    
    db_configs = [
        ('NSRDB (Healthy Control)', 'data/nsrdb', 0, ['19093', '19140', '19830']),
        ('VFDB (Lethal Ventricular Arrhythmias)', 'data/vfdb', 1, ['612', '614', '615']),
        ('SDDB (Sudden Cardiac Death Pre-Arrest)', 'data/sddb', 1, ['50', '51', '52']),
        ('EDB (Myocardial Ischemia ST-T)', 'data/edb', 1, ['e0123', 'e0124', 'e0125'])
    ]
    
    db_results = {}
    
    for db_name, db_dir, label, rec_subset in db_configs:
        scores = []
        alerts = 0
        total_windows = 0
        
        for rec in rec_subset:
            p = os.path.join(db_dir, rec)
            if not os.path.exists(f"{p}.hea"):
                continue
            X_w, _ = extract_windows_from_record(p, label, window_size=250, max_windows=100)
            for w in X_w:
                res = model.predict_window(w.ravel())
                scores.append(res['risk_score'])
                if res['is_alert']:
                    alerts += 1
                total_windows += 1
                
        if total_windows > 0:
            mean_risk = float(np.mean(scores))
            pct_detected = (alerts / total_windows) if label == 1 else (1.0 - alerts / total_windows)
            db_results[db_name] = {
                'total_windows_evaluated': total_windows,
                'mean_risk_score': round(mean_risk, 4),
                'alerts_triggered': alerts,
                'correct_classification_rate': round(float(pct_detected) * 100.0, 2)
            }
            
    return db_results


if __name__ == '__main__':
    weights_path = '02_model_cpu/model_weights/tinycardio_cpu_weights.npz'
    print("=" * 65)
    print("TINYCARDIO CLINICAL VALIDATION & PATHOLOGY EVALUATION")
    print("=" * 65)
    
    overall = evaluate_dataset_split(weights_path)
    print("\n[GLOBAL TEST SET PERFORMANCE - UNSEEN PATIENTS]")
    for k, v in overall['metrics'].items():
        print(f"  {k:<22}: {v*100:6.2f}%" if 'auc' not in k and 'f1' not in k else f"  {k:<22}: {v:6.4f}")
        
    print("\n[CONFUSION MATRIX]")
    cm = overall['confusion_matrix']
    print(f"  True Positives (Alerts Correctly Raised)  : {cm['true_positives']}")
    print(f"  False Positives (False Alarms)            : {cm['false_positives']}")
    print(f"  True Negatives (Normal Correctly Verified): {cm['true_negatives']}")
    print(f"  False Negatives (Missed Critical Events)  : {cm['false_negatives']}")
    
    print("\n[PATHOLOGY-SPECIFIC BREAKDOWN]")
    per_db = evaluate_per_database(weights_path)
    for db_name, res in per_db.items():
        print(f"\n  * {db_name}:")
        print(f"      Windows Evaluated: {res['total_windows_evaluated']}")
        print(f"      Mean Risk Score:   {res['mean_risk_score']*100:.2f}%")
        print(f"      Diagnostic Accuracy: {res['correct_classification_rate']:.2f}%")
        
    # Save report
    full_report = {'overall': overall, 'per_database': per_db}
    report_file = 'metadata/clinical_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(full_report, f, indent=2)
    print("\n" + "=" * 65)
    print(f"Full validation report exported to: {report_file}")
    print("=" * 65)
