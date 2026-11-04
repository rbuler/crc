"""Create overlays for every mask-containing slice of patient_33.

Usage:
	python a.py [results_2026/validation/no_mc] [overlay_output]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np
from PIL import Image


def load_volume(path: Path) -> np.ndarray:
	data = np.asarray(nib.load(str(path)).get_fdata())
	# A singleton/channel dimension is common in model outputs.
	data = np.squeeze(data)
	if data.ndim != 3:
		raise ValueError(f"expected a 3-D NIfTI volume, got {data.shape}")
	return data


def choose_pair(patient: Path):
	nii = list(patient.rglob("*.nii.gz"))
	results = [p for p in nii if "result" in p.name.lower()]
	images = [p for p in nii if "result" not in p.name.lower()
			  and not any(x in p.name.lower() for x in ("score", "result", "logit"))]
	if not results or not images:
		return None, None
	# Prefer an image explicitly labelled image; otherwise use the first candidate.
	images.sort(key=lambda p: ("image" not in p.name.lower(), len(str(p))))
	return images[0], results[0]


def make_overlay(image_file: Path, mask_file: Path, output_root: Path) -> None:
	image = load_volume(image_file)
	mask = load_volume(mask_file)
	if image.shape != mask.shape:
		raise ValueError(f"shape mismatch: {image.shape} vs {mask.shape}")

	result = np.nan_to_num(mask)
	foreground = result > 0
	if not foreground.any():
		raise ValueError("result contains no positive voxels")
	# Include up to 50 sagittal slices on either side of the tumour.
	positive_slices = np.flatnonzero(foreground.any(axis=(1, 2)))
	first = max(0, int(positive_slices.min()) - 50)
	last = min(foreground.shape[0] - 1, int(positive_slices.max()) + 50)
	for x in range(max(first, 50), min(last, 150) + 1):
		image_slice = image[x, :, :]
		img = np.nan_to_num(image_slice)
		result_slice = result[x, :, :]
		lo, hi = np.percentile(img, (1, 99))
		if hi <= lo:
			lo, hi = float(img.min()), float(img.max() or 1)
		img = np.clip((img - lo) / (hi - lo), 0, 1)
		output = output_root / "original" / f"slice_{x:03d}.png"
		output.parent.mkdir(parents=True, exist_ok=True)
		fig, ax = plt.subplots(figsize=(7, 7), facecolor="#10131a")
		ax.set_facecolor("#10131a")
		ax.imshow(img, cmap="gray", interpolation="nearest")
		ax.imshow(np.ma.masked_where(result_slice == 0, result_slice),
				cmap=ListedColormap(["#d62728", "#1f77b4", "#35d07f"]),
				norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3), alpha=0.55,
				interpolation="nearest")
		legend = ax.legend(
			handles=[
			Patch(facecolor="#d62728", edgecolor="none", alpha=0.75,
					 label="FP"),
			Patch(facecolor="#1f77b4", edgecolor="none", alpha=0.75,
					 label="FN"),
			Patch(facecolor="#35d07f", edgecolor="none", alpha=0.75,
					 label="TP"),

		],
			loc="lower left", frameon=True, fancybox=True, framealpha=0.9,
			facecolor="#171c26", edgecolor="#596273", labelcolor="white",
			fontsize=10, borderpad=0.8, handlelength=1.2)
		legend.get_frame().set_linewidth(0.8)
		ax.axis("off")
		fig.subplots_adjust(0, 0, 1, 1)
		fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0,
				facecolor=fig.get_facecolor())
		plt.close(fig)


def create_gif(output_root: Path) -> Path:
	"""Create a GIF from the generated slice_070.png through slice_110.png files."""
	frames = []
	for slice_index in range(70, 111):
		path = output_root / "original" / f"slice_{slice_index:03d}.png"
		if path.exists():
			with Image.open(path) as frame:
				frames.append(frame.convert("RGB"))
	if not frames:
		raise OSError("no slice images found for GIF")

	gif_path = output_root / "original" / "slices_70_110.gif"
	frames[0].save(
		gif_path,
		save_all=True,
		append_images=frames[1:],
		duration=100,
		loop=0,
	)
	return gif_path


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("input", nargs="?", default="results_2026/validation/no_mc")
	parser.add_argument("output", nargs="?", default="results_2026/validation/no_mc_overlays/patient_33")
	# Interactive environments (e.g. Jupyter/IDE consoles) may add their own
	# command-line arguments. Ignore those instead of failing to parse them.
	args, _ = parser.parse_known_args()

	root, output_root = Path(args.input), Path(args.output)
	patients = [root / "patient_33"]
	for patient in patients:
		image, mask = choose_pair(patient)
		if image is None:
			print(f"Skipping {patient}: no image/mask pair")
			continue
		try:
			make_overlay(image, mask, output_root)
			gif_path = create_gif(output_root)
			print(f"Wrote overlays to {output_root}; GIF: {gif_path}")
		except (OSError, ValueError) as exc:
			print(f"Skipping {patient}: {exc}")


if __name__ == "__main__":
	main()
