import ast
import json
import logging
import os
import random
import re
import sys

import monai.transforms as mt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import utils
import yaml
from monai.inferers import SlidingWindowInferer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset_seg import CRCDataset_seg
from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from utils import evaluate_segmentation, numpy_to_list


SPLITS = 10
PROB_THRESHOLD = 0.5
USE_MC = False
RESULTS_ROOT = "results_2026"

MODEL_PATHS = [
    "best_model_ae559da8-a149-4a42-b1f0-3d1618f79206.pth",  # 1
    "best_model_2c5cbe95-534d-4d13-b640-d06f9cb4902f.pth",  # 2
    "best_model_f6f767bb-ef12-47a3-bbb4-91e7f86901a0.pth",  # 3
    "best_model_940c745a-b772-41c6-9097-81293ba46fc2.pth",  # 4
    "best_model_63bda0be-35cd-45f4-b8c8-ba3023cb1e94.pth",  # 5
    "best_model_9fd71df9-2974-4ec6-9d9d-dbbd9770b475.pth",  # 6
    "best_model_be989c74-6ab5-4046-98be-b5d67622c2c4.pth",  # 7
    "best_model_814ccb96-72da-43c5-8551-b1b16120e0fb.pth",  # 8
    "best_model_af77cf3e-eb1e-49c9-b667-610e714719df.pth",  # 9
    "best_model_ec84797f-9555-4ca9-ae43-f34083d33df2.pth",  # 10
]

EXCLUDED_UNHEALTHY_IDS = {
    "stair_step_artifact_ids": [1, 19, 98],
    "slice_thickness_ids": [97, 128, 137],  # slice thickness > 5mm
    "colon_blockage_ids": [64, 77, 173],
    "rectal_cancer_ids": [76],
    "explicit_ids_test": [31, 32, 47, 54, 78, 109, 73, 197, 204],  # > tx t0
}


logger = logging.getLogger(__name__)
logger_radiomics = logging.getLogger("radiomics")


def configure_logging():
    logging.basicConfig(level=logging.ERROR)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_config(config_name="config.yml"):
    parser = utils.get_args_parser(config_name)
    args, _ = parser.parse_known_args()
    with open(args.config_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def get_runtime_params(config):
    root = config["dir"]["root"]
    patch_size = ast.literal_eval(config["training"]["patch_size"])
    stride = config["training"]["stride"]
    num_workers = config["training"]["num_workers"]
    num_classes = config["training"]["num_classes"]
    mode = config["training"]["mode"]
    patch_mode = config["training"]["patch_mode"]
    seed = config["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "root": root,
        "patch_size": patch_size,
        "stride": stride,
        "num_workers": num_workers,
        "num_classes": num_classes,
        "mode": mode,
        "patch_mode": patch_mode,
        "seed": seed,
        "device": device,
    }


def build_transforms(patch_size, mode):
    train_transforms = mt.Compose([mt.ToTensord(keys=["image", "mask"])])
    val_transforms = mt.Compose(
        [
            mt.CenterSpatialCropd(keys=["image", "mask", "raw_image"], roi_size=patch_size)
            if mode == "2d"
            else mt.Lambda(lambda x: x),
            mt.ToTensord(keys=["image", "mask", "raw_image"]),
        ]
    )
    return [train_transforms, val_transforms]


def load_filtered_dataframes(root, config):
    df_path_u = os.path.join(root, config["dir"]["processed"], "unhealthy_df.pkl")
    df_path_h = os.path.join(root, config["dir"]["processed"], "healthy_df.pkl")

    df_u = pd.read_pickle(df_path_u)
    df_h = pd.read_pickle(df_path_h)

    excluded_ids = (
        EXCLUDED_UNHEALTHY_IDS["stair_step_artifact_ids"]
        + EXCLUDED_UNHEALTHY_IDS["slice_thickness_ids"]
        + EXCLUDED_UNHEALTHY_IDS["colon_blockage_ids"]
        + EXCLUDED_UNHEALTHY_IDS["rectal_cancer_ids"]
    )
    explicit_ids_test = EXCLUDED_UNHEALTHY_IDS["explicit_ids_test"]

    df_u = df_u[~df_u["id"].astype(int).isin(excluded_ids)]
    df_u_explicit_test = df_u[df_u["id"].astype(int).isin(explicit_ids_test)]
    df_u = df_u[~df_u["id"].astype(int).isin(explicit_ids_test)]

    return df_u, df_h, df_u_explicit_test


def extract_t(value):
    if value is None:
        return None

    value_str = str(value).strip()
    match = re.match(r"^(T\d)", value_str, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def load_tnm_data(root, config, df_u):
    clinical_data_dir = os.path.join(root, config["dir"]["clinical_data"])
    default_missing = pd._libs.parsers.STR_NA_VALUES
    clinical_data = pd.read_excel(
        clinical_data_dir,
        index_col=False,
        na_filter=True,
        na_values=default_missing,
    )

    tnm_data = clinical_data.rename(
        columns={"TNM wg mnie": "TNM", "Nr pacjenta": "id"}
    )[["id", "TNM"]].dropna(subset=["id"])
    tnm_data = tnm_data[tnm_data["id"].astype(int).isin(df_u["id"].astype(int))]
    tnm_data["T_extracted"] = tnm_data["TNM"].apply(extract_t)

    most_frequent_t = tnm_data["T_extracted"].dropna().mode()[0]
    tnm_data["T_clean"] = tnm_data["T_extracted"].fillna(most_frequent_t)
    return tnm_data


def build_datasets(root, config, df_u, df_h, transforms, patch_size, stride, mode, patch_mode):
    dataset_u = CRCDataset_seg(
        root_dir=root,
        df=df_u,
        config=config,
        transforms=transforms,
        patch_size=patch_size,
        stride=stride,
        num_patches=100,
        mode=mode,
        patch_mode=patch_mode,
        return_raw_image=True,
    )

    dataset_h = CRCDataset_seg(
        root_dir=root,
        df=df_h,
        config=config,
        transforms=transforms,
        patch_size=patch_size,
        stride=stride,
        num_patches=100,
        mode=mode,
        patch_mode=patch_mode,
        return_raw_image=True,
    )
    return dataset_u, dataset_h


def build_folds(dataset_u, tnm_data, seed, splits=SPLITS):
    ids_train_val = dataset_u.df["id"].astype(int).unique().tolist()
    stratification_labels = tnm_data.set_index("id").reindex(ids_train_val)["T_clean"].values
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    folds = list(skf.split(ids_train_val, stratification_labels))
    return ids_train_val, folds


def enable_mc_dropout(model):
    dropout3d_layers = [m for m in model.modules() if isinstance(m, nn.Dropout3d)]
    for layer in dropout3d_layers:
        layer.train()


@torch.no_grad()
def mc_forward(model, inputs, targets, inferer, num_classes, prob_thresh, t=20, body_mask=None):
    model.eval()
    enable_mc_dropout(model)

    preds_list = []
    metrics_list = []

    for _ in range(t):
        logits = inferer(inputs=inputs, network=model)
        if body_mask is not None:
            logits[body_mask == 0] = -1e10

        preds = torch.sigmoid(logits)
        preds_list.append(preds)

        metrics = evaluate_segmentation(
            preds,
            targets,
            num_classes=num_classes,
            prob_thresh=prob_thresh,
            logits_input=False,
        )
        metrics_list.append(metrics)

    preds_stack = torch.stack(preds_list, dim=0)
    mean_pred = preds_stack.mean(dim=0)
    std_pred = preds_stack.std(dim=0)

    metrics_mean = {}
    metrics_std = {}

    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        if isinstance(values[0], np.ndarray):
            arr = np.stack(values, axis=0)
            mean_arr = arr.mean(axis=0)
            metrics_mean[key] = (mean_arr >= 0.5).astype(int)
            metrics_std[key] = arr.std(axis=0)
        else:
            metrics_mean[key] = np.mean(values)
            metrics_std[key] = np.std(values)

    return mean_pred, std_pred, metrics_mean, metrics_std


def build_model(weights_path, patch_size, device):
    model = UNETR_PP(
        in_channels=1,
        out_channels=14,
        img_size=tuple(patch_size),
        depths=[3, 3, 3, 3],
        dims=[32, 64, 128, 256],
        do_ds=False,
    )
    model.out1.conv.conv = torch.nn.Conv3d(16, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    checkpoint = torch.load(weights_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def to_numpy_safe(tensor_like):
    try:
        return tensor_like.cpu().numpy()
    except Exception:
        try:
            return tensor_like.tensor.cpu().numpy()
        except Exception:
            return torch.as_tensor(tensor_like).cpu().numpy()


def to_spatial_array(tensor_like, name, dtype=None):
    """Convert a single-volume tensor/array to a 3-D spatial array."""
    array = to_numpy_safe(tensor_like)

    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 3:
        raise ValueError(
            f"{name} must describe one 3-D volume; received shape {array.shape}."
        )

    return array.astype(dtype) if dtype is not None else array


def build_output_dir(phase_name, fold, use_mc, patient_id):
    if phase_name == "validation":
        return os.path.join(
            RESULTS_ROOT,
            "validation",
            "mc" if use_mc else "no_mc",
            f"patient_{patient_id}",
        )

    return os.path.join(
        RESULTS_ROOT,
        "test",
        f"fold_{fold}",
        "mc" if use_mc else "no_mc",
        f"patient_{patient_id}",
    )


def save_nifti_outputs(output_dir, patient_id, image, mask, final_output, logits=None, final_output_logit=None):
    os.makedirs(output_dir, exist_ok=True)

    mask_np = to_spatial_array(mask, "mask", np.uint8)
    final_output_np = to_spatial_array(final_output, "final_output", np.uint8)
    image_np = to_spatial_array(image, "image")

    if mask_np.shape != final_output_np.shape:
        raise ValueError(
            "Mask and prediction spatial shapes differ: "
            f"{mask_np.shape} versus {final_output_np.shape}."
        )

    combined_output = np.zeros_like(mask_np)
    combined_output[mask_np == 1] = 1
    combined_output[final_output_np == 1] = 2
    combined_output[(mask_np == 1) & (final_output_np == 1)] = 3

    image_nifti = nib.Nifti1Image(image_np, np.eye(4))
    mask_nifti = nib.Nifti1Image(mask_np, np.eye(4))
    combined_output_nifti = nib.Nifti1Image(combined_output, np.eye(4))

    nib.save(image_nifti, os.path.join(output_dir, f"{patient_id}_image.nii.gz"))
    nib.save(mask_nifti, os.path.join(output_dir, f"{patient_id}_mask.nii.gz"))
    nib.save(combined_output_nifti, os.path.join(output_dir, f"{patient_id}_result.nii.gz"))

    if logits is not None and final_output_logit is not None:
        logits_arr = to_spatial_array(logits, "logits", np.float32)
        scores_arr = to_spatial_array(final_output_logit, "scores", np.float32)

        nib.save(nib.Nifti1Image(logits_arr, np.eye(4)), os.path.join(output_dir, f"{patient_id}_logits.nii.gz"))
        nib.save(nib.Nifti1Image(scores_arr, np.eye(4)), os.path.join(output_dir, f"{patient_id}_scores.nii.gz"))


def run_phase(
    phase_name,
    dataloader,
    model,
    inferer,
    fold,
    num_classes,
    device,
    prob_thresh=PROB_THRESHOLD,
    use_mc=USE_MC,
):
    all_patients_metrics = {}
    dataloader.dataset.dataset.set_mode(train_mode=False)

    totals = None
    num_samples = len(dataloader)

    with torch.no_grad():
        for _, (_, _, image, mask, patient_ids, raw_image) in enumerate(dataloader):
            patient_id = str(patient_ids[0])

            inputs = image.to(device, dtype=torch.float32).unsqueeze(0)
            targets = mask["mask"].to(torch.device("cpu"), dtype=torch.long).unsqueeze(0)
            body_mask = mask["body_mask"].to(torch.device("cpu"), dtype=torch.long).unsqueeze(0)

            if use_mc:
                mean_pred, _, metrics, metrics_std = mc_forward(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    inferer=inferer,
                    num_classes=num_classes,
                    prob_thresh=prob_thresh,
                    t=20,
                    body_mask=body_mask,
                )

                metrics_str = " | ".join(
                    [f"{key}: {float(value):.3f}" for key, value in metrics.items() if "patient" not in key]
                )
                print(f"Patient_ID: {patient_id:<7} | {metrics_str}")

                metrics_std_str = " | ".join(
                    [f"{key}: {float(value):.3f}" for key, value in metrics_std.items() if "patient" not in key]
                )
                print(f"Patient_ID: {patient_id:<7} | {metrics_std_str}")

                final_output = (mean_pred.squeeze(0) > prob_thresh).cpu().numpy().astype(np.uint8)
                logits = None
                final_output_logit = None
            else:
                logits = inferer(inputs=inputs, network=model)
                logits[body_mask == 0] = -1e10

                metrics = evaluate_segmentation(
                    logits,
                    targets,
                    num_classes=num_classes,
                    prob_thresh=prob_thresh,
                )
                metrics_str = " | ".join(
                    [f"{key}: {float(value):.3f}" for key, value in metrics.items() if "patient" not in key]
                )
                print(f"Patient_ID: {patient_id:<7} | {metrics_str}")

                final_output_logit = torch.sigmoid(logits)
                final_output = (final_output_logit.squeeze(0) > prob_thresh).cpu().numpy().astype(np.uint8)

            output_dir = build_output_dir(phase_name, fold, use_mc, patient_id)
            save_nifti_outputs(
                output_dir=output_dir,
                patient_id=patient_id,
                image=raw_image,
                mask=targets,
                final_output=final_output,
                logits=logits,
                final_output_logit=final_output_logit,
            )

            if totals is None:
                totals = {key: 0 for key in metrics.keys()}
            for key, value in metrics.items():
                totals[key] += value

            all_patients_metrics[patient_id] = {
                "fold": fold,
                "metrics": metrics,
            }

    averages = {key: total / num_samples for key, total in totals.items()}
    avg_metrics_str = ", ".join(
        [f"Average {key}: {avg:.4f}" for key, avg in averages.items() if "patient" not in key]
    )
    print(f"[{phase_name}] {avg_metrics_str}")

    return all_patients_metrics


def save_phase_metrics_json(phase_name, fold, use_mc, all_patients_metrics):
    if phase_name == "validation":
        save_path = os.path.join(
            RESULTS_ROOT,
            "figures",
            "all_patients_metrics_mc.json" if use_mc else "all_patients_metrics.json",
        )
    else:
        save_path = os.path.join(
            RESULTS_ROOT,
            "figures",
            "test",
            f"fold_{fold}",
            "all_patients_metrics_mc.json" if use_mc else "all_patients_metrics.json",
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        json.dump(numpy_to_list(all_patients_metrics), file, indent=4)


def save_aggregated_json(path, payload):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            json.dump(numpy_to_list(payload), file, indent=4)
    except Exception as exc:
        print(f"Warning: failed to save aggregated JSON at {path}: {exc}")


def main():
    configure_logging()
    config = load_config("config.yml")
    runtime = get_runtime_params(config)

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    set_seed(runtime["seed"])

    transforms = build_transforms(runtime["patch_size"], runtime["mode"])
    df_u, df_h, _ = load_filtered_dataframes(runtime["root"], config)
    tnm_data = load_tnm_data(runtime["root"], config, df_u)

    dataset_u, dataset_h = build_datasets(
        root=runtime["root"],
        config=config,
        df_u=df_u,
        df_h=df_h,
        transforms=transforms,
        patch_size=runtime["patch_size"],
        stride=runtime["stride"],
        mode=runtime["mode"],
        patch_mode=runtime["patch_mode"],
    )

    ids_train_val, folds = build_folds(dataset_u, tnm_data, runtime["seed"], splits=SPLITS)

    agg_validation_all = {}
    agg_test_all = {}

    for fold, path in enumerate(MODEL_PATHS, start=1):
        for fold_idx, (_, val_idx) in enumerate(folds):
            if fold_idx + 1 != fold:
                continue

            val_ids = [ids_train_val[idx] for idx in val_idx]
            val_dataset = torch.utils.data.Subset(
                dataset_u,
                [idx for idx in range(len(dataset_u)) if int(dataset_u.get_patient_id(idx)) in val_ids],
            )
            test_dataset = torch.utils.data.Subset(dataset_h, range(len(dataset_h)))

            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=runtime["num_workers"])
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=runtime["num_workers"])

            weights_path = os.path.join(config["dir"]["root"], "models", path)
            print(fold, val_ids, weights_path)

            model = build_model(weights_path, runtime["patch_size"], runtime["device"])
            inferer = SlidingWindowInferer(
                roi_size=tuple(runtime["patch_size"]),
                sw_batch_size=36,
                overlap=0.75,
                mode="constant",
                device=torch.device("cpu"),
            )

            for phase_name, dataloader in (("validation", val_dataloader), ("test", test_dataloader)):
                all_patients_metrics = run_phase(
                    phase_name=phase_name,
                    dataloader=dataloader,
                    model=model,
                    inferer=inferer,
                    fold=fold,
                    num_classes=runtime["num_classes"],
                    device=runtime["device"],
                    prob_thresh=PROB_THRESHOLD,
                    use_mc=USE_MC,
                )

                save_phase_metrics_json(phase_name, fold, USE_MC, all_patients_metrics)

                if phase_name == "validation":
                    agg_validation_all[f"fold_{fold}"] = all_patients_metrics
                else:
                    agg_test_all[f"fold_{fold}"] = all_patients_metrics

    save_aggregated_json(
        os.path.join(RESULTS_ROOT, "figures", "validation", "all_patients_metrics_aggregated.json"),
        agg_validation_all,
    )
    save_aggregated_json(
        os.path.join(RESULTS_ROOT, "figures", "test", "all_patients_metrics_aggregated.json"),
        agg_test_all,
    )


if __name__ == "__main__":
    main()
