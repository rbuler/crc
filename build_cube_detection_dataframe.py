# %%
import os
import numpy as np
import pandas as pd
import nibabel as nib
import json
import multiprocessing
from scipy.ndimage import uniform_filter
from tqdm import tqdm

def cube_fits_in_intersection(pred_mask, gt_mask, cube_size):
    """
    Returns a tuple of booleans:
    - First value indicates if a cube of size cube_size^3 can fit entirely inside the intersection.
    - Second value indicates if a cube of size cube_size^3 can fit entirely inside the ground truth mask (gt_mask).
    """
    intersection = (pred_mask & gt_mask).astype(np.uint8)
    smoothed_intersection = uniform_filter(intersection, size=cube_size, mode='constant', origin=0) * (cube_size**3)
    fits_in_intersection = np.any(smoothed_intersection >= cube_size**3)

    smoothed_gt = uniform_filter(gt_mask.astype(np.uint8), size=cube_size, mode='constant', origin=0) * (cube_size**3)
    fits_in_gt = np.any(smoothed_gt >= cube_size**3)

    return fits_in_intersection, fits_in_gt


def find_patient_files(patient_dir):
    image_path = None
    scores_path = None
    logits_path = None
    mask_path = None
    for fname in os.listdir(patient_dir):
        fn = fname.lower()
        if fn.endswith('_image.nii.gz') or fn.endswith('_image.nii'):
            image_path = os.path.join(patient_dir, fname)
        if fn.endswith('_scores.nii.gz') or fn.endswith('_scores.nii'):
            scores_path = os.path.join(patient_dir, fname)
        if fn.endswith('_logits.nii.gz') or fn.endswith('_logits.nii'):
            logits_path = os.path.join(patient_dir, fname)
        if fn.endswith('_mask.nii.gz') or fn.endswith('_mask.nii'):
            mask_path = os.path.join(patient_dir, fname)
    return image_path, scores_path, logits_path, mask_path


def load_nifti_as_array(path):
    return nib.load(path).get_fdata()


def process_patient(patient_dir, dataset_name, fold_id, patient_id, thresholds, max_side=35):
    """Process one patient and return a single-row dict with max cube side fitted per threshold.

    - thresholds: iterable of thresholds (e.g. 0.05..0.95)
    - max_side: maximum cube side to test (inclusive). We'll try sizes 1..max_side.
    For validation: uses intersection = pred & (gt != 0).
    For test: uses pred only (gt is typically empty).
    Returns: dict with keys: dataset, fold, patient_id, and for each threshold 'th{:.2f}': int(max_side_found)
    """
    # if dataset name is 'test', we do not have ground truth masks
    if dataset_name.lower() == 'validation':
        image_path, scores_path, logits_path, mask_path = find_patient_files(patient_dir)
    else:
        image_path, scores_path, logits_path, mask_path = find_patient_files(patient_dir)
        mask_path = None  # ignore GT masks for test patients

    out = {'dataset': dataset_name, 'fold': str(fold_id), 'patient_id': str(patient_id), 'image_path': image_path, 'scores_path': scores_path}
    if scores_path is None:
        # missing scores -> fill thresholds with 0 and empty cube_detection
        cube_sizes = list(range(1, int(max_side) + 1))
        for thr in thresholds:
            out[f'th{thr:.2f}'] = 0
        out['cube_sizes'] = json.dumps(cube_sizes)
        # for each threshold, a list of zeros (no cube fits)
        empty_detection = {f'th{thr:.2f}': [0] * len(cube_sizes) for thr in thresholds}
        out['cube_detection'] = json.dumps(empty_detection)
        return out

    try:
        scores = load_nifti_as_array(scores_path)
    except Exception as e:
        print(f"Failed to load scores for {patient_dir}: {e}")
        cube_sizes = list(range(1, int(max_side) + 1))
        for thr in thresholds:
            out[f'th{thr:.2f}'] = 0
        out['cube_sizes'] = json.dumps(cube_sizes)
        empty_detection = {f'th{thr:.2f}': [0] * len(cube_sizes) for thr in thresholds}
        out['cube_detection'] = json.dumps(empty_detection)
        return out

    gt = None
    if mask_path and os.path.exists(mask_path):
        try:
            gt = load_nifti_as_array(mask_path)
        except Exception:
            gt = None

    # prepare cube sizes 1..max_side
    cube_sizes = list(range(1, int(max_side) + 1))

    # dictionary that will map threshold -> list of 0/1 indicating fit per cube size
    cube_detection = {}

    for thr in thresholds:
        pred = (scores >= thr).astype(np.uint8)
        gt = gt.astype(np.uint8) if gt is not None else None
        if gt is not None:
            intersection = (pred & (gt != 0)).astype(np.uint8)
        else:
            intersection = pred.astype(np.uint8)
        intersection = intersection.squeeze()
        # compute boolean fit for every cube size 1..max_side (1 means fits somewhere)
        fits_list = []
        max_found = 0
        for k in cube_sizes:
            sm = uniform_filter(intersection.astype(np.uint8), size=k, mode='constant') * (k ** 3)
            fits = 1 if (sm >= (k ** 3)).any() else 0
            fits_list.append(int(fits))
            if fits:
                max_found = int(k) if k > max_found else max_found

        # save max found per-threshold (existing behavior)
        out[f'th{thr:.2f}'] = int(max_found)
        # save the fits list for this threshold
        cube_detection[f'th{thr:.2f}'] = fits_list

    # store cube_sizes and cube_detection as JSON strings so they fit into a single CSV column
    out['cube_sizes'] = json.dumps(cube_sizes)
    out['cube_detection'] = json.dumps(cube_detection)

    return out


def collect_validation_patients(val_dir):
    patients = []
    if not os.path.isdir(val_dir):
        return patients
    for p in sorted(os.listdir(val_dir)):
        # only consider patient directories that start with `patient_(`
        if not p.startswith('patient_('):
            continue
        pd = os.path.join(val_dir, p)
        if os.path.isdir(pd):
            patients.append(pd)
    return patients


def collect_test_patients(test_root, folds):
    patients = []
    for f in folds:
        fold_dir = os.path.join(test_root, f'fold_{f}', 'no_mc')
        if not os.path.isdir(fold_dir):
            continue
        for p in sorted(os.listdir(fold_dir)):
            # only consider patient directories that start with `patient_(`
            if not p.startswith('patient_('):
                continue
            pd = os.path.join(fold_dir, p)
            if os.path.isdir(pd):
                patients.append((f, pd))
    return patients

# %%
# --- configuration (edit inline) -----------------------------------------
VAL_DIR = 'inference_output_last/validation/no_mc'
TEST_ROOT = 'inference_output_last/test'
FOLDS = list(range(1, 11))
OUT_CSV = 'figures/cube_detection_allpatients.csv'
# make thresholds from 0.1 to 0.9 step 0.1 but add also 0.01 0.05 0.95
THRESHOLDS = list(np.arange(0.1, 1.0, 0.1)) 
THRESHOLDS = [0.01, 0.05] + THRESHOLDS + [0.95]
MAX_SIDE = 35
# -------------------------------------------------------------------------


def process_patient_args(args):
    return process_patient(*args)


def run_parallel(arg_list, desc='processing'):
    results = []
    num_workers = max(1, (os.cpu_count() or 1) - 1)
    with multiprocessing.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_patient_args, arg_list), total=len(arg_list), desc=desc):
            results.append(res)
    return results


rows = []

val_patients = collect_validation_patients(VAL_DIR)

print(f'Found {len(val_patients)} validation patients in {VAL_DIR}')
val_args = []
for patient_dir in val_patients:
    pid = os.path.basename(patient_dir).replace('patient_', '')
    val_args.append((patient_dir, 'validation', 0, pid, THRESHOLDS, MAX_SIDE))

if val_args:
    val_results = run_parallel(val_args, desc='Validation patients')
    rows.extend(val_results)
# %%
# test
test_patients = collect_test_patients(TEST_ROOT, FOLDS)
print(f'Found {len(test_patients)} test patients across folds {FOLDS}')
test_args = []
for f, patient_dir in test_patients:
    pid = os.path.basename(patient_dir).replace('patient_', '')
    test_args.append((patient_dir, 'test', f, pid, THRESHOLDS, MAX_SIDE))

if test_args:
    test_results = run_parallel(test_args, desc='Test patients')
    rows.extend(test_results)

df = pd.DataFrame(rows)
out_csv = OUT_CSV
os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
df.to_csv(out_csv, index=False)
print(f'Wrote {len(df)} patient rows to {out_csv}')

# %%

