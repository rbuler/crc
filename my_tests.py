import numpy as np

def check_masks_inside_boxes(data_loader):
    # This function checks if the mask pixels are inside the bounding boxes
    for batch in data_loader:
        images, targets = batch["image"], batch["boxes"]
        labels, masks = batch["labels"], batch["mask"]
        for i in range(len(images)):
            mask = masks[i]
            labels = labels[i]
            boxes = targets[i]
            total_mask_pixels = np.count_nonzero(mask)
            inside_pixels = 0
            bbox_volume = 0
            for box, label in zip(boxes, labels):
                xmin, ymin, zmin, xmax, ymax, zmax = map(int, box)
                mask_region = mask[:, xmin:xmax, ymin:ymax, zmin:zmax]
                bbox_volume += (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
                inside_pixels += np.count_nonzero(mask_region == label)
            if total_mask_pixels > 0:
                percentage_inside = (inside_pixels / total_mask_pixels) * 100
                print(f"{percentage_inside:.2f}% of mask pixels are inside the bounding boxes")
                percentage_bbox_volume_occupied = (inside_pixels / bbox_volume) * 100 if bbox_volume > 0 else 0
                print(f"{percentage_bbox_volume_occupied:.2f}% of the bounding box volume is occupied by the mask")
            else:
                print(f"No mask pixels found")
                assert inside_pixels != 0 or bbox_volume != 0 or labels.sum() != 0 or boxes.sum() != 0 or mask.sum() != 0
