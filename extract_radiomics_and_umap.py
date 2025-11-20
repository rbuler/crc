# %%
import os
import sys
import argparse
import warnings
from datetime import datetime
import multiprocessing as mp
from functools import partial
import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import logging
import radiomics

logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)

try:
    from radiomics import featureextractor
except Exception:
    raise ImportError("pyradiomics package not available. Install with `pip install pyradiomics`.")

try:
    import umap
except Exception:
    raise ImportError("umap-learn is required. Install with `pip install umap-learn`.")

from sklearn.preprocessing import StandardScaler

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def find_patient_files(patient_dir):
    image_path = None
    mask_path = None
    result_path = None
    for fname in os.listdir(patient_dir):
        if fname.endswith('_image.nii') or fname.endswith('_image.nii.gz'):
            image_path = os.path.join(patient_dir, fname)
        if fname.endswith('_mask.nii') or fname.endswith('_mask.nii.gz'):
            mask_path = os.path.join(patient_dir, fname)
        if fname.endswith('_result.nii') or fname.endswith('_result.nii.gz'):
            result_path = os.path.join(patient_dir, fname)
    return image_path, mask_path, result_path


def init_worker(params):
    """Initializer for worker processes: set global extractor."""
    global EXTRACTOR
    EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(params)


def worker_extract(job):
    """Worker: job = (image_path, mask_source, meta)
    mask_source may be a numpy array OR a file path (string). The meta may contain
    'label_vals' (list/iterable of int) describing which label values to select from the mask.
    The worker will load the mask if a path is provided, build a binary mask by selecting
    those label values, and then proceed with extraction. If the selected labels are absent
    the worker returns an entry marking the missing prediction/gt.
    """
    image_path, mask_source, meta = job
    try:
        image = nib.load(image_path).get_fdata()
    except Exception as e:
        out = dict(meta)
        out['extraction_status'] = 'read_image_failed'
        out['error'] = str(e)
        return out

    # load mask if a path was supplied, otherwise assume it's an ndarray
    if isinstance(mask_source, str):
        try:
            mask_arr = nib.load(mask_source).get_fdata()
        except Exception as e:
            out = dict(meta)
            out['extraction_status'] = 'read_mask_failed'
            out['error'] = str(e)
            return out
    else:
        mask_arr = np.asarray(mask_source)
    
    
    # remove single dimensions from image and mask
    image = np.squeeze(image)
    mask_arr = np.squeeze(mask_arr)

    # select labels if provided in meta and convert to a binary mask (0/1)
    label_vals = meta.get('label_vals', None)
    if label_vals is not None:
        try:
            mask_arr = np.isin(mask_arr, list(label_vals)).astype(np.uint8)
        except Exception:
            # fallback: single-value equality or treat non-zero as mask
            if isinstance(label_vals, (list, tuple)) and len(label_vals) == 1:
                mask_arr = (mask_arr == label_vals[0]).astype(np.uint8)
            else:
                mask_arr = (mask_arr != 0).astype(np.uint8)
    else:
        mask_arr = (mask_arr != 0).astype(np.uint8)

    # now that mask_arr is binary, create SimpleITK images and ensure shape match
    if image.shape == mask_arr.shape:
        image_sitk = sitk.GetImageFromArray(image)
        mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    else:
        out = dict(meta)
        out['extraction_status'] = 'image_mask_shape_mismatch'
        return out        
    


    # if mask has no selected voxels or only voxel is 1, skip extraction and mark as missing prediction/gt
    if np.count_nonzero(mask_arr) == 0 or np.count_nonzero(mask_arr) == 1:
        out = dict(meta)
        region = meta.get('region', '')
        if region == 'val_gt':
            out['extraction_status'] = 'no_gt'
        else:
            out['extraction_status'] = 'no_pred'
        out['note'] = 'mask_empty_for_labels'
        return out

    try:
        feats = EXTRACTOR.execute(image_sitk, mask_sitk)
    except Exception as e:
        warnings.warn(f"Radiomics extraction failed for {image_path} meta={meta}: {e}")
        out = dict(meta)
        out['extraction_status'] = 'extraction_failed'
        out['error'] = str(e)
        return out

    numeric = {k: float(v) for k, v in feats.items()}
    out = dict(meta)
    out.update(numeric)
    out['extraction_status'] = 'ok'
    return out


def collect_jobs_from_validation(val_dir):
    jobs = []
    missing = []
    if not os.path.isdir(val_dir):
        print(f"Validation dir {val_dir} not found")
        return jobs, missing
    for patient in sorted(os.listdir(val_dir)):
        patient_dir = os.path.join(val_dir, patient)
        if not os.path.isdir(patient_dir):
            continue
        image_path, mask_path, result_path = find_patient_files(patient_dir)
        pid = patient.replace('patient_', '') if patient.startswith('patient_') else patient
        # GT from mask==1 — do not load mask here; let worker check label presence.
        if image_path and mask_path:
            meta = {'dataset': 'validation', 'fold': 'validation', 'patient_id': pid, 'region': 'val_gt', 'label_vals': [1]}
            # store path instead of mask array; worker will load and check labels
            jobs.append((image_path, mask_path, meta))
        else:
            # missing image or mask -> mark GT as missing
            missing.append({'dataset': 'validation', 'fold': 'validation', 'patient_id': pid, 'region': 'val_gt', 'extraction_status': 'no_image_or_mask'})

        # Pred from result: labels 2 or 3 (predicted positives) — pass path & label info
        if image_path and result_path:
            meta = {'dataset': 'validation', 'fold': 'validation', 'patient_id': pid, 'region': 'val_pred', 'label_vals': [2, 3]}
            jobs.append((image_path, result_path, meta))
        else:
            # no result file -> mark as missing pred
            missing.append({'dataset': 'validation', 'fold': 'validation', 'patient_id': pid, 'region': 'val_pred', 'extraction_status': 'no_pred'})
    return jobs, missing


def collect_jobs_from_test(test_root, folds):
    jobs = []
    missing = []
    for fold in folds:
        fold_dir = os.path.join(test_root, f'fold_{fold}', 'no_mc')
        if not os.path.isdir(fold_dir):
            continue
        for patient in sorted(os.listdir(fold_dir)):
            patient_dir = os.path.join(fold_dir, patient)
            if not os.path.isdir(patient_dir):
                continue
            image_path, mask_path, result_path = find_patient_files(patient_dir)
            pid = patient.replace('patient_', '') if patient.startswith('patient_') else patient
            # For test (healthy), use only result label == 2 (false positives)
            if image_path and result_path:
                meta = {'dataset': 'test', 'fold': f'fold_{fold}', 'patient_id': pid, 'region': 'test_fp', 'label_vals': [2]}
                jobs.append((image_path, result_path, meta))
            else:
                missing.append({'dataset': 'test', 'fold': f'fold_{fold}', 'patient_id': pid, 'region': 'test_fp', 'extraction_status': 'no_pred'})
    return jobs, missing


def main():

    val_dir = 'inference_output_last/validation/no_mc'
    test_root = 'inference_output_last/test'
    folds = list(range(1, 11))
    n_procs = max(1, os.cpu_count() - 1)
    out_csv = 'radiomics_umap_output'

    j_val, missing_val = collect_jobs_from_validation(val_dir)
    j_test, missing_test = collect_jobs_from_test(test_root, folds)
    jobs = j_val + j_test
    missing = missing_val + missing_test

    if len(jobs) == 0 and len(missing) == 0:
        print('No extraction jobs found. Exiting.')
        return

    print(f'Found {len(jobs)} extraction jobs (+ {len(missing)} missing entries). Using {n_procs} processes')

    # save jobs so we can skip collection
    joblib.dump(jobs, 'extraction_jobs.pkl')
    joblib.dump(missing, 'extraction_missing.pkl')
    

    params = 'params.yml'

    results = []
    with mp.Pool(processes=n_procs, initializer=init_worker, initargs=(params,)) as pool:
        if tqdm is not None:
            for r in tqdm(pool.imap_unordered(worker_extract, jobs), total=len(jobs), desc='Extracting'):
                results.append(r)
        else:
            for r in pool.imap_unordered(worker_extract, jobs):
                results.append(r)

    all_rows = []
    if results:
        all_rows.extend(results)
    if missing:
        all_rows.extend(missing)

    if len(all_rows) == 0:
        print('No features extracted and no missing entries to record. Exiting.')
        return

    df = pd.DataFrame(all_rows)
    # normalize and save
    df_cols_before = df.columns.tolist()
    df.rename(columns=lambda x: x.replace(' ', '_').replace('\n', '_'), inplace=True)

    time_str = datetime.now().strftime("%m-%d-%H-%M-%S")

    if out_csv:
        # if user passed a file path, use it; if passed a directory, write default filename inside
        out_path = out_csv
        if os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)
            out_csv_file = os.path.join(out_path, f'radiomics_features_extracted_{time_str}.csv')
        else:
            # assume it's a file path; ensure parent dir exists and add timestamp to filename
            parent = os.path.dirname(out_path) or '.'
            os.makedirs(parent, exist_ok=True)
            base, ext = os.path.splitext(os.path.basename(out_path))
            if ext == '':
                ext = '.csv'
            out_csv_file = os.path.join(parent, f'{base}_{time_str}{ext}')
        df.to_csv(out_csv_file, index=False)
        print('Wrote features to', out_csv_file)
    else:
        # default output path in figures with timestamp
        out_root = os.path.join('figures', f'radiomics_extract_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(out_root, exist_ok=True)
        out_csv_file = os.path.join(out_root, f'radiomics_features_extracted_{time_str}.csv')
        df.to_csv(out_csv_file, index=False)
        print('Wrote features to', out_csv_file)

    # Optionally run UMAP on numeric features and show plot
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
    if numeric_df.shape[0] >= 2:
        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df.values)
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        emb_df = df.loc[numeric_df.index].copy()
        emb_df['umap_1'] = emb[:, 0]
        emb_df['umap_2'] = emb[:, 1]

        # plot validation gt, validation pred, test pred on same plot
        plt.figure(figsize=(10, 8))
        for region, marker, color in [('val_gt', 'o', 'blue'), ('val_pred', 's', 'cyan'), ('test_fp', '^', 'orange')]:
            mask = emb_df['region'] == region
            if mask.any():
                plt.scatter(emb_df.loc[mask, 'umap_1'], emb_df.loc[mask, 'umap_2'], c=color, label=region, alpha=0.8)
        plt.legend()
        plt.title('UMAP of extracted radiomics (val_gt / val_pred / test_fp)')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.tight_layout()
        # save plot alongside the CSV output
        try:
            save_dir = os.path.dirname(out_csv_file) or '.'
            plot_path = os.path.join(save_dir, f'radiomics_umap_validation_test_{time_str}.png')
            plt.savefig(plot_path, dpi=200)
            print('Wrote UMAP plot to', plot_path)
        except Exception as e:
            print('Warning: failed to save UMAP plot:', e)
        plt.show()


if __name__ == '__main__':
    main()
# %%
