# %%
import json
import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
# %%
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
def extract_fold_detections(data):
    """Return a dict mapping fold_number -> list of patient_detection arrays.

    Handles multiple JSON shapes:
      - flat: {pid: { 'fold': int, 'metrics': {...} }}
      - nested by patient id: {pid: {subid: { 'fold':.., 'metrics': {...} }}}
      - nested by fold: {'fold_1': {pid: { 'fold':.., 'metrics': {...}}}, ...}
    """
    folds = {}
    if not isinstance(data, dict):
        return folds

    # top-level keys are like 'fold_1' ? then iterate folds->patients
    top_keys = list(data.keys())
    if all(isinstance(k, str) and k.lower().startswith('fold') for k in top_keys):
        for fkey, pdict in data.items():
            if not isinstance(pdict, dict):
                continue
            # try to infer numeric fold id
            try:
                fold_num = int(fkey.split('_')[-1])
            except Exception:
                fold_num = fkey
            for pid, pdata in pdict.items():
                if not isinstance(pdata, dict):
                    continue
                metrics = pdata.get('metrics')
                if not metrics:
                    continue
                det = metrics.get('patient_detection')
                if det is None:
                    continue
                folds.setdefault(fold_num, []).append(det)
        return folds

    # else, could be flat or nested patient dict
    sample = next(iter(data.values()))
    # flat: pid -> pdata (contains 'metrics')
    if isinstance(sample, dict) and 'metrics' in sample:
        for pid, pdata in data.items():
            if not isinstance(pdata, dict):
                continue
            metrics = pdata.get('metrics')
            if not metrics:
                continue
            det = metrics.get('patient_detection')
            if det is None:
                continue
            fold_num = pdata.get('fold', 1)
            folds.setdefault(fold_num, []).append(det)
        return folds

    # nested: pid -> {subid: pdata}
    for pid, subdict in data.items():
        if not isinstance(subdict, dict):
            continue
        for subid, pdata in subdict.items():
            if not isinstance(pdata, dict):
                continue
            metrics = pdata.get('metrics')
            if not metrics:
                continue
            det = metrics.get('patient_detection')
            if det is None:
                continue
            fold_num = pdata.get('fold', 1)
            folds.setdefault(fold_num, []).append(det)
    return folds


# build fold means and plot
folds_dict = extract_fold_detections(data)
fold_means = []
for fold, detections in folds_dict.items():
    detections = np.array(detections)
    # trim to minimal length across patients in this fold
    min_len = min(d.shape[0] for d in detections)
    detections = np.array([d[:min_len] for d in detections])
    fold_mean = detections.mean(axis=0)
    fold_means.append(fold_mean)

fold_means = np.array(fold_means)
if fold_means.size == 0:
    raise RuntimeError('No patient_detection arrays found in JSON')

mean_across_folds = fold_means.mean(axis=0)
std_across_folds = fold_means.std(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(cube_sizes[: mean_across_folds.shape[0]], mean_across_folds, color="blue", label="Mean Patient Detection")
plt.fill_between(cube_sizes[: mean_across_folds.shape[0]],
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
    # Use extract_fold_detections to support multiple JSON shapes
    folds_dict = extract_fold_detections(data)
    # Compute mean per fold, trimming to minimal length per fold
    fold_means = []
    for fold, det_list in folds_dict.items():
        det_array = np.array(det_list)
        if det_array.size == 0:
            continue
        min_len = min(d.shape[0] for d in det_array)
        det_array = np.array([d[:min_len] for d in det_array])
        fold_mean = det_array.mean(axis=0)
        fold_means.append(fold_mean)
    if len(fold_means) == 0:
        return np.zeros((0, 0))
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
plt.ylabel("Metric value", fontsize=12)
plt.ylim(0, 1)
plt.xlim(1, x_lim)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.title("Sensitivity and Specificity vs Cube Size (Mean ± STD Across Folds)",
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ----------------- ROC curve treating cube size as threshold -----------------
# At each cube size k, TPR = sensitivity_mean[k], FPR = 1 - specificity_mean[k]
fpr = 1.0 - specificity_mean
tpr = sensitivity_mean

# append (0,0) and (1,1) to complete ROC curve
fpr = np.concatenate(([0.0], fpr, [1.0]))
tpr = np.concatenate(([0.0], tpr, [1.0]))

# ensure arrays are same length and sorted by increasing FPR for AUC
order = np.argsort(fpr)
fpr_sorted = fpr[order]
tpr_sorted = tpr[order]

roc_auc = auc(fpr_sorted, tpr_sorted)

plt.figure(figsize=(6, 6))
plt.step(fpr_sorted, tpr_sorted, where='post', color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC curve across cube sizes')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# %%


csv_path = 'figures/cube_detection_allpatients.csv'  # your CSV file
all_data = pd.read_csv(csv_path)

# assign fold numbers to validation patients (robust to several JSON shapes)
with open(json_pos, 'r') as f:
    data_pos = json.load(f)

def _extract_validation_folds(d):
    folds = {}
    if not isinstance(d, dict):
        return folds
    keys = list(d.keys())
    # case: top-level keys are 'fold_1', 'fold_2', ...
    if all(isinstance(k, str) and k.lower().startswith('fold') for k in keys):
        for fkey, pdict in d.items():
            try:
                fold_num = int(fkey.split('_')[-1])
            except Exception:
                fold_num = fkey
            if not isinstance(pdict, dict):
                continue
            for pid, pdata in pdict.items():
                # pdata may be dict with 'fold' or not
                if isinstance(pdata, dict):
                    folds[str(pid)] = pdata.get('fold', fold_num)
                else:
                    folds[str(pid)] = fold_num
        return folds
    # case: flat mapping pid -> pdata (pdata contains 'fold')
    sample = next(iter(d.values()))
    if isinstance(sample, dict) and 'fold' in sample:
        for pid, pdata in d.items():
            if isinstance(pdata, dict):
                folds[str(pid)] = pdata.get('fold', 1)
        return folds
    # case: nested pid -> subid -> pdata
    for pid, sub in d.items():
        if isinstance(sub, dict):
            for subid, pdata in sub.items():
                if isinstance(pdata, dict):
                    folds[str(subid)] = pdata.get('fold', 1)
    return folds

validation_folds = _extract_validation_folds(data_pos)

# all_data patient_id may be like "('123',)", extract numeric id
def _extract_numeric_id(s):
    if not isinstance(s, str):
        return str(s)
    if s.startswith("(") and s.endswith(")"):
        try:
            inner = s[1:-1]
            inner = inner.strip("'\" ")
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1]
            return inner.split(',')[0].strip("'\" ")
        except Exception:
            return s
    return s
all_data['patient_id'] = all_data['patient_id'].apply(_extract_numeric_id)
# assign folds for validation patients
def _assign_fold(row):
    if str(row['dataset']).lower() == 'validation':
        pid = str(row['patient_id'])
        return validation_folds.get(pid, 1)
    else:
        return row.get('fold', 1)
all_data['fold'] = all_data.apply(_assign_fold, axis=1)



def _parse_cube_detection(s):
    """Robustly parse the `cube_detection` cell which may be JSON or a Python literal
    and may contain doubled quotes from CSV escaping. Returns a dict of threshold -> list.
    """
    if pd.isna(s):
        return {}
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        try:
            return dict(s)
        except Exception:
            return {}
    # try json.loads first
    try:
        return json.loads(s)
    except Exception:
        pass
    # sometimes CSV doubles quotes: replace "" with " and try again
    try:
        return json.loads(s.replace('""', '"'))
    except Exception:
        pass
    # fallback to ast.literal_eval
    try:
        return ast.literal_eval(s)
    except Exception:
        # last ditch: attempt simple replace and literal_eval
        try:
            return ast.literal_eval(s.replace('""', '"'))
        except Exception:
            return {}


def _largest_cube_for_array(arr, cube_sizes):
    a = np.asarray(arr)
    if a.size == 0:
        return 0
    # treat truthy elements (1/True) as detections
    idx = np.where(a.astype(bool))[0]
    if idx.size == 0:
        return 0
    return int(cube_sizes[idx.max()])


def summarize_by_threshold_mean_across_folds(df, cube_sizes):
    """Compute per-threshold sensitivity/specificity averaged across folds.

    Returns a dict mapping threshold -> { 'cube_sizes': array, 'sens_mean': array,
    'sens_std': array, 'spec_mean': array, 'spec_std': array, 'auc': float }
    """
    # parse all cube_detection fields lazily and find available thresholds
    sample_row = None
    for v in df['cube_detection'].values:
        parsed = _parse_cube_detection(v)
        if parsed:
            sample_row = parsed
            break
    if sample_row is None:
        raise RuntimeError('No cube_detection content parsed from CSV')

    thresholds = sorted(sample_row.keys())

    # split datasets
    df_pos = df[df['dataset'].astype(str) == 'validation']
    df_neg = df[df['dataset'].astype(str) == 'test']

    # collect largest-cube per patient per fold per threshold
    pos_by_fold = {}
    for _, row in df_pos.iterrows():
        fold = row.get('fold')
        parsed = _parse_cube_detection(row.get('cube_detection'))
        for th in thresholds:
            arr = parsed.get(th, [])
            largest = _largest_cube_for_array(arr, cube_sizes)
            pos_by_fold.setdefault(fold, {}).setdefault(th, []).append(largest)

    neg_by_fold = {}
    for _, row in df_neg.iterrows():
        fold = row.get('fold')
        parsed = _parse_cube_detection(row.get('cube_detection'))
        for th in thresholds:
            arr = parsed.get(th, [])
            largest = _largest_cube_for_array(arr, cube_sizes)
            neg_by_fold.setdefault(fold, {}).setdefault(th, []).append(largest)

    # union of folds to consider (folds present in either set)
    folds = sorted(set(list(pos_by_fold.keys()) + list(neg_by_fold.keys())))

    results = {}
    for th in thresholds:
        sens_folds = []
        spec_folds = []
        for fold in folds:
            pos_list = pos_by_fold.get(fold, {}).get(th, [])
            neg_list = neg_by_fold.get(fold, {}).get(th, [])

            # compute per-cutoff metrics for this fold
            sens = []
            spec = []
            for cutoff in cube_sizes:
                if len(pos_list) > 0:
                    tp = sum(1 for v in pos_list if v >= cutoff)
                    sens_val = tp / len(pos_list)
                else:
                    sens_val = np.nan
                if len(neg_list) > 0:
                    fp = sum(1 for v in neg_list if v >= cutoff)
                    fpr = fp / len(neg_list)
                    spec_val = 1.0 - fpr
                else:
                    spec_val = np.nan
                sens.append(sens_val)
                spec.append(spec_val)

            sens_folds.append(np.array(sens))
            spec_folds.append(np.array(spec))

        # stack and compute mean/std across folds (skip NaNs)
        sens_stack = np.vstack([s for s in sens_folds]) if len(sens_folds) > 0 else np.zeros((0, len(cube_sizes)))
        spec_stack = np.vstack([s for s in spec_folds]) if len(spec_folds) > 0 else np.zeros((0, len(cube_sizes)))

        sens_mean = np.nanmean(sens_stack, axis=0)
        sens_std = np.nanstd(sens_stack, axis=0)
        spec_mean = np.nanmean(spec_stack, axis=0)
        spec_std = np.nanstd(spec_stack, axis=0)

        # prepend cube-size 0: by definition set sensitivity(0)=1 and specificity(0)=0
        sens_mean = np.concatenate(([1.0], sens_mean))
        sens_std = np.concatenate(([0.0], sens_std))
        spec_mean = np.concatenate(([0.0], spec_mean))
        spec_std = np.concatenate(([0.0], spec_std))
        cube_sizes_out = np.concatenate(([0], cube_sizes))

        # ROC: treat cube_size cutoff as decision threshold -> FPR = 1 - spec_mean, TPR = sens_mean
        fpr = 1.0 - spec_mean
        tpr = sens_mean
        # ensure proper ordering by increasing FPR
        order = np.argsort(fpr)
        fpr_sorted = fpr[order]
        tpr_sorted = tpr[order]
        fpr_sorted = np.concatenate(([0.0], fpr_sorted, [1.0]))
        tpr_sorted = np.concatenate(([0.0], tpr_sorted, [1.0]))

        try:
            th_auc = auc(fpr_sorted, tpr_sorted)
        except Exception:
            th_auc = float('nan')

        results[th] = {
            'cube_sizes': cube_sizes_out,
            'sens_mean': sens_mean,
            'sens_std': sens_std,
            'spec_mean': spec_mean,
            'spec_std': spec_std,
            'fpr': fpr_sorted,
            'tpr': tpr_sorted,
            'auc': th_auc,
            'folds_used': folds,
        }

    return results


# run summarization and plot ROC per threshold
cube_sizes = list(range(1, int(35) + 1))
try:
    results = summarize_by_threshold_mean_across_folds(all_data, cube_sizes)
except Exception as e:
    print('Failed to summarize cube detections:', e)
    results = {}

if results:
    plt.figure(figsize=(7, 7))
    for th, res in results.items():
        plt.step(res['fpr'], res['tpr'], where='post', lw=2, label=f"{th} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC across cube-size cutoffs (per threshold) — mean across folds')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Print summary AUCs sorted
    print('Threshold AUC summary:')
    for th, res in sorted(results.items(), key=lambda x: x[1]['auc'] if not np.isnan(x[1]['auc']) else -1, reverse=True):
        print(f"  {th}: AUC={res['auc']:.4f}")

    # ----------------- Sensitivity and Specificity means for all thresholds -----------------
    plt.figure(figsize=(11, 6))
    cmap = plt.get_cmap('tab10')
    for i, (th, res) in enumerate(sorted(results.items())):
        cs = np.asarray(res['cube_sizes'])
        sens = np.asarray(res['sens_mean'])
        spec = np.asarray(res['spec_mean'])
        color = cmap(i % 10)
        # sensitivity: solid line
        plt.plot(cs, sens, color=color, linestyle='-', lw=1.8, label=f'{th} sens')
        # specificity: dashed line
        plt.plot(cs, spec, color=color, linestyle='--', lw=1.4, label=f'{th} spec')

    plt.xlabel('Cube Size (Voxels per Side)')
    plt.ylabel('Metric (mean across folds)')
    plt.title('Sensitivity (solid) and Specificity (dashed) — Means Across Folds (per threshold)')
    plt.xlim(0, x_lim)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.4)
    # make legend compact
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
    
    
# ----------------- Build DataFrame of sensitivity/specificity per threshold and cube size -----------------
    rows = []
    for th, res in results.items():
        cs = np.asarray(res['cube_sizes'])
        sens = np.asarray(res['sens_mean'])
        spec = np.asarray(res['spec_mean'])
        for i, c in enumerate(cs):
            rows.append({
                'threshold': th,
                'cube_size': int(c),
                'sensitivity_mean': float(sens[i]) if not np.isnan(sens[i]) else np.nan,
                'specificity_mean': float(spec[i]) if not np.isnan(spec[i]) else np.nan,
            })

    df_metrics = pd.DataFrame(rows)
    out_dir = os.path.join('figures')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'threshold_sens_spec_by_cube.csv')
    try:
        df_metrics.to_csv(out_csv, index=False)
        print('Wrote sensitivity/specificity table to', out_csv)
    except Exception as e:
        print('Warning: failed to write CSV:', e)

    # keep df_metrics in scope for interactive use
    # df_metrics.head()

    # ----------------- Pivot into wide tables: thresholds x cube_size -----------------
    # specificity wide table: rows=thresholds, cols=cube_size
    try:
        df_spec_wide = df_metrics.pivot(index='threshold', columns='cube_size', values='specificity_mean')
        # ensure columns sorted numeric
        df_spec_wide = df_spec_wide.reindex(sorted(df_spec_wide.columns), axis=1)
        spec_out = os.path.join(out_dir, 'threshold_specificity_by_cube_wide.csv')
        df_spec_wide.to_csv(spec_out)
        print('Wrote specificity table to', spec_out)
    except Exception as e:
        print('Warning: failed to build/write specificity wide table:', e)

    try:
        df_sens_wide = df_metrics.pivot(index='threshold', columns='cube_size', values='sensitivity_mean')
        df_sens_wide = df_sens_wide.reindex(sorted(df_sens_wide.columns), axis=1)
        sens_out = os.path.join(out_dir, 'threshold_sensitivity_by_cube_wide.csv')
        df_sens_wide.to_csv(sens_out)
        print('Wrote sensitivity table to', sens_out)
    except Exception as e:
        print('Warning: failed to build/write sensitivity wide table:', e)


# %%

