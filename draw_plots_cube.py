import json
import numpy as np
import matplotlib.pyplot as plt

# ================== USER SET PATHS ==================
json_pos = "inference_output_last/figures/validation/all_patients_metrics_aggregated.json"        # positive patients
json_neg = "inference_output_last/figures/test/all_patients_metrics_by_fold.json" # negative patients
# ====================================================

cube_sizes = np.concatenate([np.arange(1, 20), np.arange(20, 55, 5)])
x_lim = 35


# ================== USER SET PATH ==================
json_path = "inference_output_last/figures/validation/all_patients_metrics_aggregated.json"  # your JSON file
# ====================================================
with open(json_path, 'r') as f:
    data = json.load(f)

folds_dict = {}
for pid, pdata in data.items():
    fold = pdata.get("fold", 1)
    det = pdata["metrics"].get("patient_detection")
    if det is not None:
        if fold not in folds_dict:
            folds_dict[fold] = []
        folds_dict[fold].append(det)

fold_means = []
for fold, detections in folds_dict.items():
    detections = np.array(detections) 
    fold_mean = detections.mean(axis=0) 
    fold_means.append(fold_mean)

fold_means = np.array(fold_means)

mean_across_folds = fold_means.mean(axis=0)
std_across_folds = fold_means.std(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(cube_sizes, mean_across_folds, color="blue", label="Mean Patient Detection")
plt.fill_between(cube_sizes,
                 mean_across_folds - std_across_folds,
                 mean_across_folds + std_across_folds,
                 color="blue", alpha=0.3, label="±1 STD")
plt.xlabel("Cube Size (Voxels per Side)", fontsize=12)
plt.ylabel("Fraction of Patients", fontsize=12)
plt.xlim(1, x_lim)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()


# ================== USER SET PATH ==================
json_path = "inference_output_last/figures/test/all_patients_metrics_by_fold.json"  # your JSON file
# ====================================================

# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

folds_dict = {}

for pid, sub_dict in data.items():
    for subid, pdata in sub_dict.items():
        fold = pdata.get("fold", 1)
        det = pdata["metrics"].get("patient_detection")
        if det is not None:
            if fold not in folds_dict:
                folds_dict[fold] = []
            folds_dict[fold].append(det)

fold_means = []
for fold, detections in folds_dict.items():
    detections = np.array(detections) 
    fold_mean = detections.mean(axis=0) 
    fold_means.append(fold_mean)

fold_means = np.array(fold_means)
mean_across_folds = fold_means.mean(axis=0)
std_across_folds = fold_means.std(axis=0)


plt.figure(figsize=(10, 6))
plt.plot(cube_sizes, mean_across_folds, color="orange", label="Mean Patient Detection")
plt.fill_between(cube_sizes,
                 mean_across_folds - std_across_folds,
                 mean_across_folds + std_across_folds,
                 color="orange", alpha=0.3, label="±1 STD")
plt.xlabel("Cube Size (Voxels per Side)", fontsize=12)
plt.ylabel("Fraction of Patients", fontsize=12)
plt.ylim(0, 1)
plt.xlim(1, x_lim)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()


def fold_level_means(data, nested=False):
    folds_dict = {}
    for pid, pdata in data.items():
        if nested:
            for subid, subpdata in pdata.items():
                fold = subpdata.get("fold", 1)
                det = subpdata["metrics"].get("patient_detection")
                if det is not None:
                    if fold not in folds_dict:
                        folds_dict[fold] = []
                    folds_dict[fold].append(det)
        else:
            fold = pdata.get("fold", 1)
            det = pdata["metrics"].get("patient_detection")
            if det is not None:
                if fold not in folds_dict:
                    folds_dict[fold] = []
                folds_dict[fold].append(det)
    
    # Compute mean per fold
    fold_means = []
    for fold, det_list in folds_dict.items():
        det_array = np.array(det_list)  # shape: (n_patients_in_fold, n_cube_sizes)
        fold_mean = det_array.mean(axis=0)
        fold_means.append(fold_mean)
    return np.array(fold_means)  # shape: (n_folds, n_cube_sizes)


# ----------------- LOAD JSONs -----------------
with open(json_pos, 'r') as f:
    data_pos = json.load(f)
fold_means_pos = fold_level_means(data_pos, nested=False)
sensitivity_mean = fold_means_pos.mean(axis=0)
sensitivity_std = fold_means_pos.std(axis=0)

with open(json_neg, 'r') as f:
    data_neg = json.load(f)
fold_means_neg = fold_level_means(data_neg, nested=True)
specificity_mean = 1 - fold_means_neg.mean(axis=0)
specificity_std = fold_means_neg.std(axis=0)

n_cube_sizes = sensitivity_mean.shape[0]

plt.figure(figsize=(10,6))
plt.plot(cube_sizes, sensitivity_mean, color='blue', label='Sensitivity (TPR)')
plt.fill_between(cube_sizes, sensitivity_mean - sensitivity_std, sensitivity_mean + sensitivity_std,
                 color='blue', alpha=0.2)
plt.plot(cube_sizes, specificity_mean, color='green', label='Specificity (TNR)')
plt.fill_between(cube_sizes, specificity_mean - specificity_std, specificity_mean + specificity_std,
                 color='green', alpha=0.2)

plt.xlabel("Cube Size (Voxels per Side)", fontsize=12)
plt.ylabel("Fraction of Patients", fontsize=12)
plt.ylim(0, 1)
plt.xlim(1, x_lim)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.title("Sensitivity and Specificity vs Cube Size (Mean ± STD Across Folds)",
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()